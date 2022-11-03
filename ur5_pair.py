from setup import DEFAULT_ORN, DEFAULT_ORN_LEFT, DEFAULT_ORN_RIGHT, get_ur5s
from time import sleep
import numpy as np


class UR5Pair:
    def __init__(self,
                 ur5s=get_ur5s(),
                 gui=False):
        self.left_ur5, self.right_ur5 = ur5s
        self.gui = gui

    def get_pose(self):
        return self.left_ur5.get_pose(), self.right_ur5.get_pose()

    def move_delta(self, deltas, control_ori=False, verbose=True):
        left_deltas, right_deltas = deltas
        if not control_ori: 
            left_deltas.extend([0,0,0])
            right_deltas.extend([0,0,0])
        left_deltas, right_deltas = np.array(left_deltas), np.array(right_deltas)

        left_pose, right_pose = self.get_pose()
        new_left_pose = np.array(left_pose).copy()
        new_right_pose = np.array(right_pose).copy()
    
        new_left_pose += left_deltas
        new_right_pose += right_deltas
    
        if verbose:
            print("left pose", left_pose)
            print("right pose", right_pose)
            print("new_left_pose", new_left_pose)
            print("new_right_pose", new_right_pose)
            input("Confirm movement?")

        ur5_pair.move(
            move_type="l",
            params=[
                # left
                new_left_pose,
                # right
                new_right_pose],
            blocking=True,
            use_pos=True)


    def all_ur5s_reached_target(self):
        return self.left_ur5.reached_target() and\
            self.right_ur5.reached_target()

    def homej(self, blocking=True, **kwargs):
        kwargs['blocking'] = False
        self.left_ur5.homej(**kwargs)
        self.right_ur5.homej(**kwargs)
        if blocking:
            # wait until both reaches
            while not self.all_ur5s_reached_target():
                # IO so scheduler prioritizes process
                print('\r ', end='')
                sleep(0.05)

    def movej(self, params, blocking=True, **kwargs):
        kwargs['blocking'] = False
        self.left_ur5.movej(params=params[0], **kwargs)
        self.right_ur5.movej(params=params[1], **kwargs)
        if blocking:
            # wait until both reaches
            while not self.all_ur5s_reached_target():
                # IO so scheduler prioritizes process
                print('\r ', end='')
                sleep(0.01)
        return True

    def movel(self, params, blocking=True, **kwargs):
        kwargs['blocking'] = False
        self.left_ur5.movel(params=params[0], **kwargs)
        self.right_ur5.movel(params=params[1], **kwargs)
        if blocking:
            # wait until both reaches
            while not self.all_ur5s_reached_target():
                # IO so scheduler prioritizes process
                print('\r ', end='')
                sleep(0.01)
        return True

    def move(self, move_type, params, blocking=True, **kwargs):
        kwargs['blocking'] = False
        self.left_ur5.move(
            move_type=move_type,
            params=params[0], **kwargs)
        self.right_ur5.move(
            move_type=move_type,
            params=params[1], **kwargs)
        if blocking:
            # wait until both reaches
            while not self.all_ur5s_reached_target():
                # IO so scheduler prioritizes process
                print('\r ', end='')
                sleep(0.01)
        return True

    def close_grippers(self, blocking=True, **kwargs):
        self.left_ur5.gripper.close(blocking=False, **kwargs)
        self.right_ur5.gripper.close(blocking=False, **kwargs)
        if blocking:
            sleep(1)

    def open_grippers(self, blocking=True, **kwargs):
        self.left_ur5.gripper.open(blocking=False, **kwargs)
        self.right_ur5.gripper.open(blocking=False, **kwargs)
        if blocking:
            sleep(1)

    def out_of_the_way(self):
        self.movel(
            params=[[0.1, 0.4, 0.3] + DEFAULT_ORN]*2,
            blocking=True,
            use_pos=True)
