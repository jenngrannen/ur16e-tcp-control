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
from gym import Env, spaces
sys.path.append(os.getcwd())


import numpy as np

print(sys.path)
from setup import DEFAULT_ORN_LEFT, DEFAULT_ORN_RIGHT, WORKSPACE_SURFACE, DIST_UR5
from base_transforms import LEFT_TRANSFORM_VIEW_TO_BASE, LEFT_TRANSLATION_VIEW_TO_BASE, RIGHT_TRANSFORM_VIEW_TO_BASE, RIGHT_TRANSLATION_VIEW_TO_BASE
from ur5_pair import UR5Pair

class Lipa1DEnv(Env):
    def __init__(self, ur_pair=None, workspace=[[0.365, 0.663]], reset_pos=None, control_ori=False):
        if ur_pair is None:
            self.ur_pair = UR5Pair()
        else:
            self.ur_pair = ur_pair
        self.workspace = workspace # list of axis limits

        self.goal_position = 0.53

        if reset_pos is not None:
            self.reset_pos = reset_pos
        else:
            left_reset_pos = [0.60408, -0.62524, 0.440529, -0.004274, -1.035806, 2.954933]
            # left_reset_pos = [0.53, -0.62524, 0.440529, -0.004274, -1.035806, 2.954933]
            right_reset_pos = [-0.467998, 0.41326, 0.32809, -0.29323, -0.9106905, 1.4417393]
            self.reset_pos = [left_reset_pos, right_reset_pos]

        self.observation_space = spaces.Box(low = np.array([workspace[0][0]]),
                                            high = np.array([workspace[0][1]]))

        self.action_space = spaces.Box(low = np.array([-1]),
                                        high = np.array([1]))

        # for pytorch_sac:
        self._max_episode_steps = 10

    def seed(self, seed):
        pass

    def reset(self, rescale_needed=True):
        print("RESETTING")
        self.ur_pair.move(
            move_type="l",
            params=self.reset_pos,
            blocking=True,
            use_pos=True)
        obs = self._get_obs(rescale_needed=rescale_needed)
        return obs

    def step(self, action, rescale_needed=True, verbose=False):
        # assumes 1d action input

        # rescale action (assuming input is normalized)
        rescaled_action = self._rescale_action(action) if rescale_needed else action
        left_pose_base, right_pose_base = self.ur_pair.get_pose()
        # convert base frame to world frame
        left_pose_world, right_pose_world = self._base_to_world(left_pose_base, right_pose_base)

        new_left_pose_world = np.array(left_pose_world).copy()
        new_right_pose_world = np.array(right_pose_world).copy()

        left_deltas = np.zeros(len(left_pose_world))
        left_deltas[0] = rescaled_action[0]
        new_left_pose_world += left_deltas

        # convert from world frame to base frame
        new_left_pose_base, new_right_pose_base = self._world_to_base(new_left_pose_world, new_right_pose_world)
        # JENN: I think this should be clipping for world?
        new_left_x = np.clip([new_left_pose_base[0]], [self.workspace[0][0]], [self.workspace[0][1]])[0]
        new_left_pose_base[0] = new_left_x
        # assert new_right_pose_base == right_pose_base

        if verbose:
            print("---WORLD/VIEW FRAME---")
            print("current left", left_pose_world)
            print("new left", new_left_pose_world)
            print("current right", right_pose_world)
            print("new right", new_right_pose_world)
            print()

            print("---BASE FRAME---")
            print("current left", left_pose_base)
            print("new left", new_left_pose_base)
            print("current right", right_pose_base)
            print("new right", new_right_pose_base)

            input("Confirm movement? Press any key.")

        self.ur_pair.move(
            move_type="l",
            params=[new_left_pose_base, new_right_pose_base],
            blocking=True,
            use_pos=True)

        # get obs after action
        obs = self._get_obs(rescale_needed=rescale_needed)

        # define task-specific reward outside of this class
        reward = -1*((obs[0]-self.goal_position)**2)
        # define task-specific done outside of this class
        done = reward > -0.0004 # should be within 0.02 of goal
        print("obs, reward, done:", obs, reward, done)
        if done: print("SUCCESS")

        return obs, reward, done, {}

    def _world_to_base(self, left_pose, right_pose):
        new_left_position = np.dot(LEFT_TRANSFORM_VIEW_TO_BASE, left_pose[:3]) + LEFT_TRANSLATION_VIEW_TO_BASE
        new_right_position = np.dot(RIGHT_TRANSFORM_VIEW_TO_BASE, right_pose[:3]) + RIGHT_TRANSLATION_VIEW_TO_BASE

        new_left_pose = np.array(left_pose)
        new_left_pose[:3] = new_left_position
        new_right_pose = np.array(right_pose)
        new_right_pose[:3] = new_right_position
        return new_left_pose, new_right_pose


    def _base_to_world(self, left_pose, right_pose):
        new_left_position = np.dot(np.linalg.inv(LEFT_TRANSFORM_VIEW_TO_BASE), (left_pose[:3] - LEFT_TRANSLATION_VIEW_TO_BASE))
        new_right_position = np.dot(np.linalg.inv(RIGHT_TRANSFORM_VIEW_TO_BASE), (right_pose[:3] - RIGHT_TRANSLATION_VIEW_TO_BASE))

        new_left_pose = np.array(left_pose)
        new_left_pose[:3] = new_left_position
        new_right_pose = np.array(right_pose)
        new_right_pose[:3] = new_right_position
        return new_left_pose, new_right_pose


    def _get_obs(self, rescale_needed=True):
        # assume observation space is just 1d of robot ee pos
        final_left_pose, final_right_pose = self.ur_pair.get_pose()

        # change this from robot base frame to world frame
        world_left_pose, world_right_pose = self._base_to_world(final_left_pose, final_right_pose)

        obs = final_left_pose[0] # JENN: shouldn't this be world_left_pose? idk
        # print("world pose", world_left_pose, final_left_pose)
        # obs = self._rescale_obs(obs) if rescale_needed else obs
        return np.array([obs])

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
        x_scale = 0.1
        return [action[0]*x_scale]

    def render(self):
        return np.zeros((64,64,3))

if __name__ == "__main__":
    ur_pair = UR5Pair()
    # workspace = [ # these are made up examples and need to be calibrated
    #     [-10, 10], # x lim
    #     [-2, 3], # y lim
    #     [-5, -2] # z lim
    # ]

    workspace = [ 
        [0.365, 0.663], # x lim
    ]
    robot_env = Lipa1DEnv(ur_pair, workspace)
    print("robot_env", robot_env.ur_pair.get_pose()[0][:3])
    robot_env.reset()
    while True:
        action = [0.5]
        robot_env.step(action, verbose=True)
