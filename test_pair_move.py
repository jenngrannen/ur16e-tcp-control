import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from setup import DEFAULT_ORN_LEFT, DEFAULT_ORN_RIGHT, WORKSPACE_SURFACE, DIST_UR5
from ur5_pair import UR5Pair


def test_move(ur5_pair, front_camera, height: float, grasp_width: float,
            max_grasp_width=0.6):
    # from .setup import DEFAULT_ORN, DIST_UR5
    while True:
        left_pose, right_pose = ur5_pair.get_pose()
        print("left pose", left_pose)
        print("right pose", right_pose)

        grasp_width += 0.02
        # dx = (DIST_UR5 - grasp_width)/2
        dx = 0.00
        height = -0.05

        control_ori = False
        deltas = [0,0,height]
        if not control_ori:
            deltas.extend([0,0,0])
        deltas = np.array(deltas)

        new_left_pose = np.array(left_pose).copy()
        new_right_pose = np.array(right_pose).copy()
        new_left_pose += deltas
        new_right_pose += deltas
        print("new_left_pose", new_left_pose)
        print("new_right_pose", new_right_pose)
        input("Confirm movement? Press any key.")

        ur5_pair.move(
            move_type="l",
            params=[
                # left
                new_left_pose,
                # [0.67844598, -0.16828214,  0.49778534 + 0.1] + DEFAULT_ORN_LEFT,
                # [dx, 0, height] + DEFAULT_ORN_LEFT,
                # right
                # [dx, 0, height] + DEFAULT_ORN],
                # [0, 0, 0] + DEFAULT_ORN_RIGHT],
                new_right_pose],
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
