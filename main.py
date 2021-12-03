import pickle


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    with open('/home/nrogovoy/Downloads/mt_trajectory.pkl', 'rb') as f:
        data = pickle.load(f)
    print_hi('PyCharm')

