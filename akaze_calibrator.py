import logging
import os
from typing import List, Tuple, Dict

import cv2
import numpy as np
import plotly.graph_objs as plotly_go
from scipy import linalg

from structs import Frame, AbstractCalibrator


class InvalidFeature(Exception):
    pass


def compute_pose_matrix(points_in_left: np.ndarray, points_in_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper link https://web.stanford.edu/class/cs273/refs/umeyama.pdf
    Args:
        points_in_left:
        points_in_right:

    Returns:

    """
    num_points = len(points_in_left)
    dim_points = len(points_in_left[0])
    # cursory check that the number of points is sufficient
    if num_points < dim_points:
        raise ValueError('Number of points must be greater/equal {0}.'.format(dim_points))

    # construct matrices out of the two point sets for easy manipulation
    left_mat = np.array(points_in_left).T
    right_mat = np.array(points_in_right).T

    # center both data sets on the mean
    left_mean = left_mat.mean(axis=1)
    right_mean = right_mat.mean(axis=1)
    left_M = left_mat - np.tile(left_mean, (num_points, 1)).T
    right_M = right_mat - np.tile(right_mean, (num_points, 1)).T

    M = left_M.dot(right_M.T)
    U, S, Vt = linalg.svd(M)
    V = Vt.T

    # V * diag(1,1,det(U*V)) * U' - diagonal matrix ensures that we have a rotation and not a reflection
    R = V.dot(np.diag((1, 1, linalg.det(U.dot(V))))).dot(U.T)
    t = right_mean - R.dot(left_mean)
    return R, t


def ratio_test(matches: List[List[cv2.DMatch]], matching_ratio_thresh: float) -> List[cv2.DMatch]:
    good_matches = []
    for match1, match2 in matches:
        if match1.distance < matching_ratio_thresh * match2.distance:
            good_matches.append(match1)
    return good_matches


def calc_XYZ(p_left: np.ndarray, p_right: np.ndarray,
             f: float, cx: float, cy: float, T: float) -> Tuple[float, float, float]:
    disparity = np.abs(p_left[0] - p_right[0])
    depth = (f * T) / (-disparity)
    Z = depth
    X = -((p_left[0] - cx) * Z) / f
    Y = ((p_left[1] - cy) * Z) / f
    return X, Y, Z


def find_good_matches(descriptors_left: np.ndarray, descriptors_right: np.ndarray,
                      matching_ratio_thresh: float) -> List[cv2.DMatch]:
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    # match features
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)
    good_matches = ratio_test(matches, matching_ratio_thresh)
    return good_matches


class AkazeCalibrator(AbstractCalibrator):
    model_name = 'Akaze-Ransac'

    def __init__(self,
                 matching_ratio_thresh_internal=0.8,
                 matching_ratio_thresh_extrenal=0.8,
                 disparity_y_diff_threshold=8,
                 ransac_min_samples_to_fit=3,
                 ransac_iterations=1000,
                 ransac_inlier_threshold=200,
                 min_points_per_frame=20):
        super().__init__()
        self.matching_ratio_thresh_internal = matching_ratio_thresh_internal  # lower value means more aggressive filtering
        self.matching_ratio_thresh_extrenal = matching_ratio_thresh_extrenal  # lower value means more aggressive filtering
        self.disparity_y_diff_threshold = disparity_y_diff_threshold
        self.ransac_min_samples_to_fit = ransac_min_samples_to_fit
        self.ransac_iterations = ransac_iterations
        self.ransac_inlier_threshold = ransac_inlier_threshold
        self.inliers_percentage = 0.0
        self.min_points_per_frame = min_points_per_frame

    def akaze_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(image, None)
        descriptors = descriptors.astype(np.float32)
        return keypoints, descriptors

    def akaze_get_feature_points(self, frame: Frame) -> Tuple[np.ndarray, np.ndarray]:

        # find features
        keypoints_left, descriptors_left = self.akaze_features(frame.left_image)
        keypoints_right, descriptors_right = self.akaze_features(frame.right_image)

        good_matches = find_good_matches(descriptors_left, descriptors_right,
                                         self.matching_ratio_thresh_internal)
        if os.environ.get('CREATE_DEBUG_DATA', ''):
            flann_matches = cv2.drawMatches(frame.left_image, keypoints_left, frame.right_image, keypoints_right,
                                            good_matches, None)
            save_path = frame.path.replace(".png", "_flann.png")
            cv2.imwrite(save_path, flann_matches)

        # calc X,Y,Z
        points, descriptors = [], []
        points_2d = []  # needed if CREATE_DEBUG_DATA='1' (in other words - if --no-debug-data not used)
        for m in good_matches:
            xy_left = np.array(keypoints_left[m.queryIdx].pt)
            xy_right = np.array(keypoints_right[m.trainIdx].pt)
            if np.abs(xy_left - xy_right)[1] <= self.disparity_y_diff_threshold:
                X, Y, Z = calc_XYZ(xy_left, xy_right, frame.f, frame.cx, frame.cy, frame.T)
                points.append((X, Y, Z))
                points_2d.append(keypoints_left[
                                     m.queryIdx])  # needed if CREATE_DEBUG_DATA='1' (in other words - if --no-debug-data not used)
                descriptors.append(descriptors_left[m.queryIdx])
            else:
                continue
        if os.environ.get('CREATE_DEBUG_DATA', ''):
            return np.array(points), np.array(descriptors), points_2d
        else:
            return np.array(points), np.array(descriptors)

    def match_source_and_target(self, points_src: np.ndarray, descriptors_src: np.ndarray,
                                points_trg: np.ndarray, descriptors_trg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        good_matches = find_good_matches(descriptors_src, descriptors_trg,
                                         self.matching_ratio_thresh_extrenal)
        matched_pts_src, matched_pts_trg = [], []
        for m in good_matches:
            matched_pts_src.append(points_src[m.queryIdx])
            matched_pts_trg.append(points_trg[m.trainIdx])
        if os.environ.get('CREATE_DEBUG_DATA', ''):
            return np.array(matched_pts_src), np.array(matched_pts_trg), good_matches
        else:
            return np.array(matched_pts_src), np.array(matched_pts_trg)

    def match_two_cameras(self, source: List[Frame], target: List[Frame]):
        pts_src, pts_trg = [], []
        for src, trg in zip(source, target):
            if os.environ.get('CREATE_DEBUG_DATA', ''):
                points_src, descriptors_src, left_2d_src = self.akaze_get_feature_points(src)
                points_trg, descriptors_trg, left_2d_tg = self.akaze_get_feature_points(trg)
                matched_src_pts, matched_trg_pts, good_matches = self.match_source_and_target(points_src,
                                                                                              descriptors_src,
                                                                                              points_trg,
                                                                                              descriptors_trg)
                flann_matches = cv2.drawMatches(src.left_image, left_2d_src, trg.left_image, left_2d_tg,
                                                good_matches, None)

                save_path = src.path.replace(".png",
                                             "{}_{}_.png".format(src.path.split("/")[-3], trg.path.split("/")[-3]))
                cv2.imwrite(save_path, flann_matches)
            else:
                # find features src
                points_src, descriptors_src = self.akaze_get_feature_points(src)

                # find features trg
                points_trg, descriptors_trg = self.akaze_get_feature_points(trg)

                # match
                matched_src_pts, matched_trg_pts = self.match_source_and_target(points_src, descriptors_src,
                                                                                points_trg, descriptors_trg)

            if matched_src_pts.shape[0] < self.min_points_per_frame:
                logging.warning("Skipping pose: too few matched points : {}".format(matched_src_pts.shape[0]))
                continue
            pts_src.append(matched_src_pts)
            pts_trg.append(matched_trg_pts)
        return pts_src, pts_trg

    def ransac(self, points_src: np.ndarray, pts_trg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed=178)
        max_inliers = 0
        pts_src = np.atleast_2d(points_src)
        pts_trg = np.atleast_2d(pts_trg)
        inliers = np.zeros(pts_src.shape[0], dtype=np.bool)

        I = np.zeros(pts_src.shape[0], dtype=np.bool)
        I[:self.ransac_min_samples_to_fit] = True
        for i in range(self.ransac_iterations):
            np.random.shuffle(I)
            R, t = compute_pose_matrix(pts_src[I], pts_trg[I])
            temp_transformed_pts_src = pts_src.dot(R.T) + t
            temp_distances = linalg.norm(temp_transformed_pts_src - pts_trg, axis=-1)
            temp_inliers = temp_distances <= self.ransac_inlier_threshold
            if temp_inliers.sum() > max_inliers:
                max_inliers = temp_inliers.sum()
                inliers = temp_inliers

        inliers_count = inliers.sum()
        if inliers_count < 3:
            raise RuntimeError("RANSAC: Could not find transformation")
        R, t = compute_pose_matrix(pts_src[inliers], pts_trg[inliers])
        return R, t

    def calculate_quality(self, pts_src: np.ndarray, pts_trg: np.ndarray, pose_index: np.ndarray) -> Dict:
        transformed_pts_src = pts_src.dot(self.R.T) + self.t
        distances = np.linalg.norm(transformed_pts_src - pts_trg, axis=-1)
        mae = np.mean(distances[distances <= self.ransac_inlier_threshold])
        max_mae = -1.0
        for inx in np.unique(pose_index):
            I = np.logical_and(pose_index == inx, distances <= self.ransac_inlier_threshold)
            max_mae = max(max_mae, np.mean(distances[I]))
        inliners = distances < self.ransac_inlier_threshold
        return {"mae": float(mae),
                "max_mae": float(max_mae),
                "inliners_ratio": float(np.mean(inliners)),
                "inliners_count": int(np.sum(inliners)),
                "distances": distances.tolist(),
                "pose_index": pose_index.tolist(),
                "ransac_inlier_threshold": self.ransac_inlier_threshold}

    def calibrate_two_cameras(self, source: List[Frame], target: List[Frame]):
        pts_src, pts_trg = self.match_two_cameras(source, target)
        pose_index = []
        for i, pts in enumerate(pts_src):
            pose_index.append(np.ones(pts.shape[0]) * (i + 1))
        pose_index = np.concatenate(pose_index)
        pts_src = np.vstack(pts_src)
        pts_trg = np.vstack(pts_trg)
        R, t = self.ransac(pts_src, pts_trg)
        if os.environ.get('CREATE_DEBUG_DATA', ''):
            vis_fig = self.__create_ransac_visualizations(pts_src, pts_trg, R, t)
            points_save_path = source[0].path.split("/")[:-3]
            points_save_path = os.path.join("/".join(points_save_path),
                                            "{}_{}_points.html".format(source[0].camera_name, target[0].camera_name))
            vis_fig.write_html(points_save_path)
        self.set_transformation(R, t, scale=1.0)
        return self.calculate_quality(pts_src, pts_trg, pose_index)

    def quality_two_cameras(self, source: List[Frame], target: List[Frame]):
        pts_src, pts_trg = self.match_two_cameras(source, target)
        pose_index = []
        for i, pts in enumerate(pts_src):
            pose_index.append(np.ones(pts.shape[0]) * (i + 1))
        pose_index = np.concatenate(pose_index)
        pts_src = np.vstack(pts_src)
        pts_trg = np.vstack(pts_trg)
        return self.calculate_quality(pts_src, pts_trg, pose_index)

    ## Debug methods
    def __draw_3d_points_single(self, data, fig=None, name=None):
        if fig is None:
            fig = plotly_go.Figure()
        color = f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})'
        data = np.array(data).squeeze()
        fig.add_trace(
            plotly_go.Scatter3d(
                x=-data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                name=name,
                text=data[:, -1],
                mode='markers', marker=dict(color=color,
                                            size=5))
        )
        return fig

    def __create_ransac_visualizations(self, pts_src: np.ndarray, pts_trg: np.ndarray, R: np.ndarray,
                                       t: np.ndarray) -> plotly_go.Figure:
        transformed_pts = pts_src.dot(R.T) + t
        vis_fig = plotly_go.Figure()
        vis_fig = self.__draw_3d_points_single(pts_trg, vis_fig, "pts_trg")
        vis_fig = self.__draw_3d_points_single(pts_src, vis_fig, "pts_src")
        vis_fig = self.__draw_3d_points_single(transformed_pts, vis_fig, "transformed_pts")
        return vis_fig
