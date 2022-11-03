import sys
import os 
sys.path.append(os.getcwd())

# from real_world.kinect import KinectClient
# from real_world.realsense import RealSense
from realur5 import UR5
# from real_world.wsg50 import WSG50
# from real_world.rg2 import RG2

# DEFAULT_ORN = [2.22, 2.22, 0.0]
DEFAULT_ORN = [2.53, -1.585, 0.0]
DEFAULT_ORN_LEFT = [0.27494682,  1.04809644, -2.41703083]
DEFAULT_ORN_RIGHT = [2.9, -1.5, 0.0]
DIST_UR5 = 1.34
WORKSPACE_SURFACE = -0.15
MIN_GRASP_WIDTH = 0.25
MAX_GRASP_WIDTH = 0.6
MIN_UR5_BASE_SAFETY_RADIUS = 0.3
# workspace pixel crop
WS_PC = [30, -165, 385, -370]

UR5_VELOCITY = 0.5
# UR5_VELOCITY = 0.2
UR5_ACCELERATION = 0.3
# UR5_ACCELERATION = 0.1

CLOTHS_DATASET = {
    'hannes_tshirt': {
        'flatten_area': 0.0524761,
        'cloth_size': (0.45, 0.55),
        'mass': 0.2
    },
}
CURRENT_CLOTH = 'hannes_tshirt'


def get_ur5s():
    return [
        UR5(name="02_robot",
            tcp_ip='192.168.0.2',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            # gripper=RG2(tcp_ip='XXX.XXX.X.XXX')),
            gripper=None),
        UR5(name="03_robot",
            tcp_ip='192.168.0.3',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            # gripper=WSG50(tcp_ip='XXX.XXX.X.XXX')),
            gripper=None),
    ]


def get_top_cam():
    return None
    # return KinectClient()


def get_front_cam():
    return None
    # return RealSense(
    #     tcp_ip='127.0.0.1',
    #     tcp_port=12345,
    #     im_h=720,
    #     im_w=1280,
    #     max_depth=3.0)

if __name__ == "__main__":
    right_arm, left_arm = get_ur5s()
    print("GOT ARMS:", right_arm, left_arm)
    right_arm