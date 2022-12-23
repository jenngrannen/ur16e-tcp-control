import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from setup import DEFAULT_ORN_LEFT, DEFAULT_ORN_RIGHT, WORKSPACE_SURFACE, DIST_UR5
from ur5_pair import UR5Pair
from base_transforms import LEFT_TRANSFORM_VIEW_TO_BASE, LEFT_TRANSLATION_VIEW_TO_BASE, RIGHT_TRANSFORM_VIEW_TO_BASE, RIGHT_TRANSLATION_VIEW_TO_BASE


def world_to_base(left_pose, right_pose):
    new_left_position = np.dot(LEFT_TRANSFORM_VIEW_TO_BASE, left_pose[:3]) + LEFT_TRANSLATION_VIEW_TO_BASE
    new_right_position = np.dot(RIGHT_TRANSFORM_VIEW_TO_BASE, right_pose[:3]) + RIGHT_TRANSLATION_VIEW_TO_BASE

    new_left_pose = np.array(left_pose)
    new_left_pose[:3] = new_left_position
    new_right_pose = np.array(right_pose)
    new_right_pose[:3] = new_right_position
    return new_left_pose, new_right_pose


def base_to_world(left_pose, right_pose):
    new_left_position = np.dot(np.linalg.inv(LEFT_TRANSFORM_VIEW_TO_BASE), (left_pose[:3] - LEFT_TRANSLATION_VIEW_TO_BASE))
    new_right_position = np.dot(np.linalg.inv(RIGHT_TRANSFORM_VIEW_TO_BASE), (right_pose[:3] - RIGHT_TRANSLATION_VIEW_TO_BASE))

    new_left_pose = np.array(left_pose)
    new_left_pose[:3] = new_left_position
    new_right_pose = np.array(right_pose)
    new_right_pose[:3] = new_right_position
    return new_left_pose, new_right_pose

def test_move(ur5_pair, front_camera, height: float, grasp_width: float,
            max_grasp_width=0.6):
    # from .setup import DEFAULT_ORN, DIST_UR5
    while True:
        left_pose_base, right_pose_base = ur5_pair.get_pose()
        # print("left pose base", left_pose)
        # print("right pose base", right_pose)

        # convert base to world
        left_pose_world, right_pose_world = base_to_world(left_pose_base, right_pose_base)
        print("left pose world", left_pose_world)
        print("right pose world", right_pose_world)

        grasp_width += 0.02
        # dx = (DIST_UR5 - grasp_width)/2
        dy = 0.0
        # height = +0.05
        height = +0.0

        control_ori = False
        # deltas = [0,0,height]
        deltas = [0, 0, 0.02]
        if not control_ori:
            deltas.extend([0,0,0])
        deltas = np.array(deltas)

        new_left_pose_world = np.array(left_pose_world).copy()
        new_right_pose_world = np.array(right_pose_world).copy()
        new_left_pose_world += deltas
        # new_right_pose_world += deltas
        print("new_left_pose world", new_left_pose_world)
        print("new_right_pose world", new_right_pose_world)
        input("Confirm movement? Press any key.")

        new_left_pose_base, new_right_pose_base = world_to_base(new_left_pose_world, new_right_pose_world)

        ur5_pair.move(
            move_type="l",
            params=[
                # left
                new_left_pose_base,
                # [0.67844598, -0.16828214,  0.49778534 + 0.1] + DEFAULT_ORN_LEFT,
                # [dx, 0, height] + DEFAULT_ORN_LEFT,
                # right
                # [dx, 0, height] + DEFAULT_ORN],
                # [0, 0, 0] + DEFAULT_ORN_RIGHT],
                new_right_pose_base],
            blocking=True,
            use_pos=True)

if __name__ == "__main__":
    ur5_pair = UR5Pair()
    print("Connected to robots", ur5_pair)
    test_move(ur5_pair, None, 0, 0)
    # import socket
    # import time

    # HOST = "192.168.0.2"
    # PORT = 30003
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((HOST, PORT))

    # s.send(("set_digital_out(0, True)"+"\n").encode("utf8"))
    # time.sleep(0.2)
    # s.send(("set_digital_out(0, False)"+"\n").encode("utf8"))

#air hockey table xmin 0.365
#air hockey table xmax 0.663