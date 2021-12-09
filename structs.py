from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Dict, MutableSet

import numpy as np


@dataclass
class Frame:
    f: float
    cx: float
    cy: float
    T: float
    left_image: np.array
    right_image: np.array
    timestamp: float


class Pair:
    def __init__(self, source_name: str, target_name: str, source_frames: List[Frame], target_frames: List[Frame]):
        self.source_name = source_name
        self.target_name = target_name
        self.source_frames = source_frames
        self.target_frames = target_frames
        self.size = len(source_frames)


@dataclass
class ExtractRecordDataStructure:
    svo_path: str

    @classmethod
    def from_dict(cls, data):
        return cls(svo_path=data['svo_path'])


@dataclass
class InputDataStructure:
    extract: List[ExtractRecordDataStructure]

    @classmethod
    def from_dict(cls, data):
        return cls(extract=[ExtractRecordDataStructure.from_dict(record) for record in data['extract']])


@dataclass
class CameraModel:
    serial_number: int
    camera_model: int
    firmware_version: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serial_number": self.serial_number,
            "camera_model": self.camera_model,
            "firmware_version": self.firmware_version,
        }


@dataclass
class CameraEyeCalibration:
    cx: float
    cy: float
    d_fov: float
    h_fov: float
    v_fov: float
    disto: List[int]
    fx: float
    fy: float
    image_size: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "d_fov": self.d_fov,
            "h_fov": self.h_fov,
            "v_fov": self.v_fov,
            "disto": self.disto,
            "fx": self.fx,
            "fy": self.fy,
            "image_size": self.image_size
        }


@dataclass
class CameraInfoCalibration:
    R: List[float]
    T: List[float]
    left_cam: CameraEyeCalibration
    right_cam: CameraEyeCalibration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "R": self.R,
            "T": self.T,
            "left_cam": self.left_cam.to_dict(),
            "right_cam": self.right_cam.to_dict(),
        }


@dataclass
class CameraSettings:
    brightness: int
    contrast: int
    exposure: int
    gain: int
    gamma: int
    hue: int
    saturation: int
    sharpness: int
    white_balance_temperature: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "exposure": self.exposure,
            "gain": self.gain,
            "gamma": self.gamma,
            "hue": self.hue,
            "saturation": self.saturation,
            "sharpness": self.sharpness,
            "white_balance_temperature": self.white_balance_temperature
        }


@dataclass
class CameraInfo:
    model: CameraModel
    calibration: CameraInfoCalibration
    calibration_raw: CameraInfoCalibration
    fps: int
    camera_setting: CameraSettings
    camera_name: int
    camera_position: str
    camera_side: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "calibration": self.calibration.to_dict(),
            "calibration_raw": self.calibration_raw.to_dict(),
            "fps": self.fps,
            "camera_setting": self.camera_setting.to_dict(),
            "camera_name": self.camera_name,
            "camera_position": self.camera_position,
            "camera_side": self.camera_side,
        }

    def __eq__(self, other_obj: 'CameraInfo') -> bool:
        if not isinstance(other_obj, CameraInfo):
            return False
        if self.camera_name == other_obj.camera_name:
            return True
        return False

    def __hash__(self) -> int:
        return self.camera_name


@dataclass
class Camera:
    serial_number: str
    rig_side: str
    rig_position: str
    settings: CameraSettings
    info: CameraInfo
    frames: List[Frame] = field(default_factory=list)

    def __eq__(self, other_obj: 'Camera') -> bool:
        if not isinstance(other_obj, Camera):
            return False
        if self.serial_number == other_obj.serial_number:
            self.frames.extend(other_obj.frames)
            return True
        return False

    def __hash__(self) -> int:
        return self.serial_number


@dataclass
class CalibrationCollection:
    rig_side: str
    cameras: List[Camera] = field(default_factory=list)


@dataclass
class Transformation:
    rotation: List[List[float]] = field(default_factory=list)
    translation: List[float] = field(default_factory=list)
    scale: float = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rotation": self.rotation,
            "translation": self.translation,
            "scale": self.scale
        }


@dataclass
class Quality:
    mae: float = None
    max_mae: float = None
    inliners_ratio: float = None
    inliners_count: int = None
    distances: List[float] = field(default_factory=list)
    pose_index: List[float] = field(default_factory=list)
    ransac_inlier_threshold: int = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae": self.mae,
            "max_mae": self.max_mae,
            "inliners_ratio": self.inliners_ratio,
            "inliners_count": self.inliners_count,
            "distances": self.distances,
            "pose_index": self.pose_index,
            "ransac_inlier_threshold": self.ransac_inlier_threshold
        }


@dataclass
class CalibrationTransformationsDataStructure:
    source_id: int
    target_id: int
    source2target: Transformation = field(default_factory=Transformation)
    target2source: Transformation = field(default_factory=Transformation)
    message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source2target": self.source2target.to_dict(),
            "target2source": self.target2source.to_dict()
        }


@dataclass
class CalibrationQualitiesDataStructure:
    source_id: int
    target_id: int
    source2target: Quality = field(default_factory=Quality)
    target2source: Quality = field(default_factory=Quality)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source2target": self.source2target.to_dict(),
            "target2source": self.target2source.to_dict()
        }


@dataclass
class CalibrationsDataStructure:
    transformations: List[CalibrationTransformationsDataStructure] = field(default_factory=list)
    qualities: List[CalibrationQualitiesDataStructure] = field(default_factory=list)
    cameras: MutableSet[Camera] = field(default_factory=set)
    took: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformations": [transformation.to_dict() for transformation in self.transformations],
            "qualities": [qualitie.to_dict() for qualitie in self.qualities],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "success": self.success,
            "took": self.took
        }

    @property
    def success(self):
        if not self.transformations:
            return False
        if all(tr.message == '' for tr in self.transformations):
            return True
        return False


class AbstractCalibrator:
    model_name = ''

    @abstractmethod
    def __init__(self):
        self.R = np.eye(3, 3)
        self.t = np.zeros(3)
        self.scale = 1.0

    def set_transformation(self, R: np.ndarray, t: np.ndarray, scale=1.0):
        self.R = R
        self.t = t
        self.scale = scale

    @abstractmethod
    def calibrate_two_cameras(self, source: List[Frame], target: List[Frame]):
        quality = {}
        return quality

    @abstractmethod
    def quality_two_cameras(self, source: List[Frame], target: List[Frame]):
        quality = {}
        return quality

    def calibrate(self, pair: Pair):
        transformation = CalibrationTransformationsDataStructure(
            source_id=pair.source_name,
            target_id=pair.target_name
        )
        quality = CalibrationQualitiesDataStructure(
            source_id=pair.source_name,
            target_id=pair.target_name
        )
        quality_data = self.calibrate_two_cameras(pair.source_frames,
                                                  pair.target_frames)
        quality.source2target = Quality(
            mae=quality_data["mae"],
            max_mae=quality_data["max_mae"],
            inliners_ratio=quality_data["inliners_ratio"],
            inliners_count=quality_data["inliners_count"],
            distances=quality_data["distances"],
            pose_index=quality_data["pose_index"],
            ransac_inlier_threshold=quality_data["ransac_inlier_threshold"]
        )
        transformation.source2target = Transformation(
            rotation=self.R.tolist(),
            translation=self.t.tolist(),
            scale=self.scale
        )
        quality_data = self.calibrate_two_cameras(pair.target_frames,
                                                  pair.source_frames)
        quality.target2source = Quality(
            mae=quality_data["mae"],
            max_mae=quality_data["max_mae"],
            inliners_ratio=quality_data["inliners_ratio"],
            inliners_count=quality_data["inliners_count"],
            distances=quality_data["distances"],
            pose_index=quality_data["pose_index"],
            ransac_inlier_threshold=quality_data["ransac_inlier_threshold"]
        )
        transformation.target2source = Transformation(
            rotation=self.R.tolist(),
            translation=self.t.tolist(),
            scale=self.scale
        )
        return transformation, quality