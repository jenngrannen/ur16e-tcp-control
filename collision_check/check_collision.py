import pybullet as p
from scipy.spatial.transform import Rotation as R
import os
import numpy as np

right_initial_positions = {
        'shoulder_pan_joint': -1.5690622952052096, 'shoulder_lift_joint': -1.5446774605904932, 'elbow_joint': 1.343946009733127,
        'wrist_1_joint': -1.3708613585093699, 'wrist_2_joint': -1.5707970583733368, 'wrist_3_joint': 0.0009377758247187636
    }

left_initial_positions = {
        'shoulder_pan_joint': -1.5690622952052096, 'shoulder_lift_joint': -1.5446774605904932, 'elbow_joint': 1.343946009733127,
        'wrist_1_joint': -1.3708613585093699, 'wrist_2_joint': -1.5707970583733368, 'wrist_3_joint': 0.0009377758247187636
    }

control_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

class URPairSim:
    def __init__(self, real_ur_pair=None, control_ori=False, vis=True):
        self.left_base_position = (0.0, -0.3, 1.025) # these are estimations
        self.right_base_position = (0.0, 0.3, 1.025)
        self.left_base_orientation = (0.9659258, 0, 0, 0.258819)
        self.right_base_orientation = (0.9659258, 0, 0, -0.258819)

        self.right_initial_positions = right_initial_positions
        self.left_initial_positions = left_initial_positions
        self._right_joint_name_to_ids = {}
        self._left_joint_name_to_ids = {}
        self._control_joint_names = control_joint_names
        self.end_eff_idx = 6

        self.control_ori = control_ori
        self.vis = vis

        if self.vis:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setPhysicsEngineParameter()
        sim_timestep = 1.0 / 240
        p.setTimeStep(sim_timestep)
        p.setGravity(0, 0, -9.8)
        self.init_robots()
        for j in range(300):
            if j % 100 == 0: print("here")
            p.stepSimulation()

    def init_robots(self):
        self.right_robot_id = p.loadURDF(os.path.join("/Users/jennifergrannen/Documents/Stanford/iliad/stable_bimanual/ur16e-tcp-control/collision_check/", "ur16e_bullet.urdf"),
                                   basePosition=self.right_base_position, useFixedBase=True,
                                   globalScaling=1.0,
                                   baseOrientation=self.right_base_orientation,
                                   physicsClientId=self._physics_client_id) # stabilize this one

        self.left_robot_id = p.loadURDF(os.path.join("/Users/jennifergrannen/Documents/Stanford/iliad/stable_bimanual/ur16e-tcp-control/collision_check/", "ur16e_bullet.urdf"),
                                   basePosition=self.left_base_position, useFixedBase=True,
                                   globalScaling=1.0,
                                   # baseOrientation=(0, 0, 1, 0),
                                   baseOrientation=self.left_base_orientation,
                                   physicsClientId=self._physics_client_id)

        assert self.right_robot_id is not None, "Failed to load the right robot model"
        assert self.left_robot_id is not None, "Failed to load the left robot model"

        # right reset
        # reset joints to home position
        num_joints = p.getNumJoints(self.right_robot_id, physicsClientId=self._physics_client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.right_robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.right_initial_positions.keys()
                self._right_joint_name_to_ids[joint_name] = i
                p.resetJointState(self.right_robot_id, i, self.right_initial_positions[joint_name], physicsClientId=self._physics_client_id)
                p.setJointMotorControl2(self.right_robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=self.right_initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)

        # left reset
        # reset joints to home position
        num_joints = p.getNumJoints(self.left_robot_id, physicsClientId=self._physics_client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.left_robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.left_initial_positions.keys()
                self._left_joint_name_to_ids[joint_name] = i
                p.resetJointState(self.left_robot_id, i, self.left_initial_positions[joint_name], physicsClientId=self._physics_client_id)
                p.setJointMotorControl2(self.left_robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=self.left_initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)


    def set_j_pos(self, robot_id, joint_pos):
        num_joints = p.getNumJoints(robot_id, physicsClientId=self._physics_client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                joint_pos_id = self._control_joint_names.index(joint_name)
                p.resetJointState(robot_id, i, joint_pos[joint_pos_id], physicsClientId=self._physics_client_id)
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=joint_pos[joint_pos_id],
                                        positionGain=0.2, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)

    def check_collision(self, actions, current_j_pos=None, current_ee_pos=None):
        # this will check if there will be a collision in a list of sequential actions
        # if there will be a collision, it will return the index of the colliding action
        if current_j_pos is not None:
            left_jpos, right_jpos = current_j_pos
            self.set_j_pos(self.left_robot_id, left_jpos)
            self.set_j_pos(self.right_robot_id, right_jpos)

        if current_ee_pos is not None:
            left_ee_quat = p.getLinkState(self.left_robot_id, self.end_eff_idx, computeLinkVelocity=1,
                                   computeForwardKinematics=1, physicsClientId=self._physics_client_id)[1]
            right_ee_quat = p.getLinkState(self.right_robot_id, self.end_eff_idx, computeLinkVelocity=1,
                                   computeForwardKinematics=1, physicsClientId=self._physics_client_id)[1]

            left_jointPoses = p.calculateInverseKinematics(self.left_robot_id, self.end_eff_idx, current_ee_pos[:3], left_ee_quat,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)
            right_jointPoses = p.calculateInverseKinematics(self.right_robot_id, self.end_eff_idx, current_ee_pos[3:], right_ee_quat,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)

            self.set_j_pos(self.left_robot_id, left_jointPoses)
            self.set_j_pos(self.right_robot_id, right_jointPoses)

        left_ee_pos, left_ee_quat = p.getLinkState(self.left_robot_id, self.end_eff_idx, computeLinkVelocity=1,
                               computeForwardKinematics=1, physicsClientId=self._physics_client_id)[0:2]
        right_ee_pos, right_ee_quat = p.getLinkState(self.right_robot_id, self.end_eff_idx, computeLinkVelocity=1,
                               computeForwardKinematics=1, physicsClientId=self._physics_client_id)[0:2]
        left_ee_pos, right_ee_pos = np.array(left_ee_pos), np.array(right_ee_pos)
        left_ee_quat, right_ee_quat = np.array(left_ee_quat), np.array(right_ee_quat)

        new_left_pose_world, new_right_pose_world = np.zeros(6), np.zeros(6)
        new_left_pose_world[0:3] = left_ee_pos
        new_left_pose_world[3:] = R.from_quat(left_ee_quat).as_rotvec()
        new_right_pose_world[0:3] = right_ee_pos
        new_right_pose_world[3:] = R.from_quat(right_ee_quat).as_rotvec()

        for action_i, single_action in enumerate(actions): # action should be a list of actions
            if not self.control_ori:
                left_deltas = np.zeros(len(new_left_pose_world))
                left_deltas[:3] = single_action[:3]
                right_deltas = np.zeros(len(new_right_pose_world))
                right_deltas[:3] = single_action[3:]

                new_left_pose_world += left_deltas
                new_right_pose_world += right_deltas
            else:
                new_left_pose_world += single_action[:(len(single_action)//2)]
                new_right_pose_world += single_action[(len(single_action)//2):]

            left_quat = R.from_rotvec(np.array(new_left_pose_world[3:])).as_quat()
            right_quat = R.from_rotvec(np.array(new_right_pose_world[3:])).as_quat()
            left_jointPoses = p.calculateInverseKinematics(self.left_robot_id, self.end_eff_idx, new_left_pose_world[:3], left_quat,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)
            right_jointPoses = p.calculateInverseKinematics(self.right_robot_id, self.end_eff_idx, new_right_pose_world[:3], right_quat,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)
            self.set_j_pos(self.left_robot_id, left_jointPoses)
            self.set_j_pos(self.right_robot_id, right_jointPoses)

            check_contacts_time = 200 if self.vis else 1 # slower rollout if visualizing
            for j in range(check_contacts_time): # check contacts doesn't work without this
                p.stepSimulation()

            if self._check_touch():
                print("Collision Detected! Terminating.")
                return action_i

            new_left_pose_world = new_left_pose_world.copy()
            new_right_pose_world = new_right_pose_world.copy()

        return None # if no collisions return None

    def _check_touch(self):
        num_joints = p.getNumJoints(self.right_robot_id, physicsClientId=self._physics_client_id)
        for i in range(num_joints):
            p0 = p.getContactPoints(self.right_robot_id, self.left_robot_id, linkIndexB=i, physicsClientId=self._physics_client_id)
            p1 = p.getContactPoints(self.left_robot_id, self.right_robot_id, linkIndexB=i, physicsClientId=self._physics_client_id)
            if not len(p0) == 0 or not len(p1) == 0:
                return True
        return False

if __name__ == "__main__":
    ur_pair_check = URPairSim()
    actions = np.array([[0, 1, 0, 0, -1, 0] for _ in range(100)])*0.01
     # if reading current real jpos from some ur_pair
    # left_jpos = self.ur_pair.left_ur5.state.get_j_pos()
    # right_jpos = self.ur_pair.right_ur5.state.get_j_pos()
    ur_pair_check.check_collision(actions, current_j_pos=None, current_ee_pos=[0.6, -0.3, 0.5, 0.6, 0.3, 0.5])
    # ur_pair_check.check_collision(actions, current_ee_pos=None, current_j_pos=[np.zeros(6),np.zeros(6)])
