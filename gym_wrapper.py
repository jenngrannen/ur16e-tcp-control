"""
This is a wrapper for taking an action from RL training and executing it on the robot.
 - It abstracts away the workspace dims, but requires a reset position to be passed in.
 - A reward and done should be defined outside this class, as this class should not contain
   task-specific information.
 - This should be only for moving the robot and also collecting sensor data
   (eg a camera or force data)
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from setup import DEFAULT_ORN_LEFT, DEFAULT_ORN_RIGHT, WORKSPACE_SURFACE, DIST_UR5
from ur5_pair import UR5Pair

class URPairEnv:
    def __init__(self, ur_pair, workspace, reset_pos=None, control_ori=False):
        self.ur_pair = ur_pair
        self.workspace = workspace # list of axis limits
        self.control_ori = control_ori
        self.angle_workspace = [
            [-np.pi, np.pi], # x angle
            [-np.pi, np.pi], # y angle
            [-np.pi, np.pi] # z angle
        ]
        if reset_pos is not None:
            self.reset_pos = reset_pos
        else:
            left_reset_pos = [0.67844598, -0.16828214,  0.49778534] + [0.27494682,  1.04809644, -2.41703083]
            right_reset_pos = [0, 0, 0] + [2.9, -1.5, 0.0] # check this
            self.reset_pos = [left_reset_pos, right_reset_pos]

    def reset(self):
        self.ur_pair.move(
            move_type="l",
            params=self.reset_pos,
            blocking=True,
            use_pos=True)
        obs = self._get_obs(rescale_needed=rescale_needed)
        return obs

    def step(self, action, rescale_needed=True, verbose=True):
        # if not self.control_ori: assumes 6d action input --> dx, dy, dz for each arm
        # if self.control_ori: assumes 12d action input

        # rescale action (assuming input is normalized)
        rescaled_action = self._convert_action(action) if rescale_needed else action
        left_pose, right_pose = self.ur_pair.get_pose()
        new_left_pose = np.array(left_pose).copy()
        new_right_pose = np.array(right_pose).copy()

        if not self.control_ori:
            left_deltas = np.zeros(len(left_pose)//2)
            left_deltas[:3] = rescaled_action[:3]
            right_deltas = np.zeros(len(right_pose)//2)
            right_deltas[:3] = rescaled_action[3:]

            new_left_pose += left_deltas
            new_right_pose += right_deltas
        else:
            new_left_pose += rescaled_action[:(len(rescaled_action)//2)]
            new_right_pose += rescaled_action[(len(rescaled_action)//2):]

        if verbose:
            print("new_left_pose", new_left_pose)
            print("new_right_pose", new_right_pose)
            input("Confirm movement? Press any key.")

        self.ur_pair.move(
            move_type="l",
            params=[new_left_pose, new_right_pose],
            blocking=True,
            use_pos=True)

        # get obs after action
        obs = self._get_obs(rescale_needed=rescale_needed)

        # define task-specific reward outside of this class
        reward = None
        # define task-specific done outside of this class
        done = False

        return obs, reward, done, {}

    def get_rgb_image(self):
        pass # todo

    def get_force_info(self):
        pass # todo

    def _get_obs(self, rescale_needed=True):
        # assume observation space is just robot ee position data
        final_left_pose, final_right_pose = self.ur_pair.get_pose()
        obs = final_left_pose
        obs.extend(final_right_pose)
        obs = _rescale_obs(obs) if rescale_needed else obs
        return obs

    def _rescale_obs(self, obs):
        # from workspace dims to -1, 1
        x_scale = max(self.workspace[0]) - min(self.workspace[0])
        y_scale = max(self.workspace[1]) - min(self.workspace[1])
        z_scale = max(self.workspace[2]) - min(self.workspace[2])
        angle_x_scale = max(self.angle_workspace[0]) - min(self.angle_workspace[0])
        angle_y_scale = max(self.angle_workspace[1]) - min(self.angle_workspace[1])
        angle_z_scale = max(self.angle_workspace[2]) - min(self.angle_workspace[2])

        x_min, y_min, z_min = min(self.workspace[0]), min(self.workspace[1]), min(self.workspace[2])
        angle_x_min, angle_y_min, angle_z_min = min(self.angle_workspace[0]), min(self.angle_workspace[1]), min(self.angle_workspace[2])

        assert len(obs) == 12
        new_obs = np.zeros(12)
        for i in range(2):
            new_obs[i*6+0] = (obs[i*6+0]-x_min)/x_min
            new_obs[i*6+1] = (obs[i*6+1]-y_min)/y_min
            new_obs[i*6+2] = (obs[i*6+2]-z_min)/z_min
            new_obs[i*6+3] = (obs[i*6+3]-angle_x_min)/angle_x_scale
            new_obs[i*6+4] = (obs[i*6+4]-angle_y_min)/angle_y_scale
            new_obs[i*6+5] = (obs[i*6+5]-angle_z_min)/angle_z_scale
        return new_obs

    def _rescale_action(self, action):
        # from -1, 1 to workspace dims
        x_scale = 0.04
        y_scale = 0.04
        z_scale = 0.04
        d_angle = np.pi/20 # calibrate this

        if not self.control_ori:
            assert len(action) == 6
            new_action = np.zeros(6)
            for i in range(2):
                new_action[i*3+0] = action[i*3+0]*x_scale
                new_action[i*3+1] = action[i*3+1]*y_scale
                new_action[i*3+2] = action[i*3+2]*z_scale
        else:
            assert len(action) == 12
            new_action = np.zeros(12)
            for i in range(2):
                new_action[i*6+0] = action[i*6+0]*x_scale
                new_action[i*6+1] = action[i*6+1]*y_scale
                new_action[i*6+2] = action[i*6+2]*z_scale
                new_action[i*6+3] = action[i*6+3]*d_angle
                new_action[i*6+4] = action[i*6+4]*d_angle
                new_action[i*6+5] = action[i*6+5]*d_angle
        return new_action




if __name__ == "__main__":
    ur_pair = UR5Pair()
    workspace = [ # these are made up examples and need to be calibrated
        [-10, 10], # x lim
        [-2, 3], # y lim
        [-5, -2] # z lim
    ]
    robot_env = URPairEnv(ur_pair, workspace)
    
