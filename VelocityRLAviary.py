import numpy as np
import pybullet as p
import torch
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType


class VelocityRLAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 target_pos: np.ndarray=np.array([1, 1, 1]),
                 flying_bound_radius: np.ndarray=np.array([4, 4, 4]),
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 24,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL
                ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = target_pos
        self.LEADER_TARGET_BUFFER = 0
        self.FOLLOWER_TARGET_BUFFER = 0.15
        self.TARGET_FEEDBACK_BUFFER_SIZE = int(ctrl_freq // 2)
        self.target_feedback_buffer = deque(maxlen=self.TARGET_FEEDBACK_BUFFER_SIZE)
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                        )
        
        self.EPISODE_LEN_SEC = 8
        self.FLYING_BOUND_RADIUS = flying_bound_radius
        self.FOLLOW_END_THRESHOLD = 0.2
        self.LEADER_TARGET_IMPORTANCE = 4

        for _ in range(self.TARGET_FEEDBACK_BUFFER_SIZE):
            self.target_feedback_buffer.append(np.zeros((self.NUM_DRONES, 12)))

    ################################################################################
    
    def _observationSpace(self):
        """
        Returns the observation space of the environment.

        Each drone is given the following as observations:
            camera_image: A (48, 64, 4) vector of the RGBA camera image taken by the drone
            kinematics: A (12,) vector describing the kinematics of the drone
            target_kinematics: A (12,) vector describing the most recent kinematics of the target the drone is following
            target_kinematics_history: A (12 * self.TARGET_FEEDBACK_BUFFER_SIZE,) vector describing the 
                                       previous kinematics of the target the drone is following over the 
                                       last self.TARGET_FEEDBACK_BUFFER_SIZE steps.
            target_pos_buffer: A (1,) vector describing the buffer distance between the drone and the target position
                               that the drone should aim to hover around

        Leader drone navigates to a static goal. target_kinematics + target_kinematics_history in this case is unchanging
        over the course of the episode.

        Follower drones navigates to the leader drone's position. target_kinematics + target_kinematics_history changes as
        the leader drone navigates to the goal.

        target_kinematics_history is used to predict the target_kinematics value. This accounts for cases when
        target_kinematics may be inaccurate due to noise or unintentional/intentional disturbance.
        """
        camera_image_obs_space = spaces.Box(low=0, 
                                            high=255, 
                                            shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), 
                                            dtype=np.uint8)
    
        # Kinematics observation vector: X Y Z R P Y VX VY VZ WX WY WZ
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo,lo,0,lo,lo,lo,lo,lo,lo,lo,lo,lo] for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for _ in range(self.NUM_DRONES)])

        kinematics_obs_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        target_history_lower_bound = np.array([[] for _ in range(self.NUM_DRONES)])
        target_history_upper_bound = np.array([[] for _ in range(self.NUM_DRONES)])

        for _ in range(self.TARGET_FEEDBACK_BUFFER_SIZE):
            target_history_lower_bound = np.hstack([target_history_lower_bound, np.array([[lo,lo,0,lo,lo,lo,lo,lo,lo,lo,lo,lo] for _ in range(self.NUM_DRONES)])])
            target_history_upper_bound = np.hstack([target_history_upper_bound, np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for _ in range(self.NUM_DRONES)])])

        target_kinematics_history_obs_space = spaces.Box(low=target_history_lower_bound, high=target_history_upper_bound, dtype=np.float32)

        target_pos_buffer_obs_space = spaces.Box(low=lo,
                                                 high=hi,
                                                 shape=(self.NUM_DRONES,), 
                                                 dtype=np.float32)

        observation_space = spaces.Dict({
                "camera_image": camera_image_obs_space, 
                "kinematics": kinematics_obs_space,
                "target_kinematics": kinematics_obs_space,
                "target_kinematics_history": target_kinematics_history_obs_space,
                "target_pos_buffer": target_pos_buffer_obs_space
            }, 
            seed=42)

        return observation_space
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        obs = {}

        # Get RGBA camera image
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
            for i in range(self.NUM_DRONES):
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i, segmentation=False)
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb[i],
                                      path=self.ONBOARD_IMG_PATH + "drone_"+str(i),
                                      frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ))
                    
        obs["camera_image"] = torch.from_numpy(np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32'))
    
        # Get drone kinematics
        drone_kinematics = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            drone_state = self._getDroneStateVector(i)
            drone_kinematics[i, :] = np.hstack([drone_state[0:3], drone_state[7:10], drone_state[10:13], drone_state[13:16]]).reshape(12,)

        obs["kinematics"] = torch.from_numpy(drone_kinematics.astype('float32'))

        # Get target object kinematics
        target_kinematics = np.zeros((self.NUM_DRONES, 12))
        target_kinematics[0] = np.hstack([self.TARGET_POS, np.zeros(9)]).reshape(12,)
        target_kinematics[1:] = np.array([drone_kinematics[0] for _ in range(self.NUM_DRONES - 1)])

        obs["target_kinematics"] = torch.from_numpy(target_kinematics.astype('float32'))

        # Get target object's kinematics history
        target_feedback_history = np.array([[] for _ in range(self.NUM_DRONES)])
        for i in range(self.TARGET_FEEDBACK_BUFFER_SIZE):
            target_feedback_history = np.hstack([target_feedback_history, self.target_feedback_buffer[i]])

        obs["target_kinematics_history"] = torch.from_numpy(target_feedback_history.astype('float32'))

        # Get buffer distance between drone and target object
        target_pos_buffer = np.zeros(self.NUM_DRONES)
        target_pos_buffer[0] = self.LEADER_TARGET_BUFFER
        target_pos_buffer[1:] = self.FOLLOWER_TARGET_BUFFER
        obs["target_pos_buffer"] = torch.tensor(target_pos_buffer)

        # Add target obect kinematics to history buffer
        self.target_feedback_buffer.append(target_kinematics)

        return obs        

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        leader_pos = states[0][0:3]
        leader_dist_to_goal = np.linalg.norm(self.TARGET_POS - leader_pos)
        # leader_reward = 1 / ((leader_dist_to_goal + 0.000001)**3)
        leader_reward = 50 - leader_dist_to_goal**3

        followers_reward = 0
        if (self.NUM_DRONES > 1):
            for i in range(self.NUM_DRONES)[1:]:
                follower_dist_to_leader = np.linalg.norm(leader_pos - states[i][0:3])
                # followers_reward += 1 / ((abs(self.IDEAL_FOLLOW_DISTANCE - follower_dist_to_leader) + 0.000001)**3)
                followers_reward += 50 - follower_dist_to_leader**3

        leader_reward_multiplier = (self.NUM_DRONES - 1) if self.NUM_DRONES > 1 else 1

        return (leader_reward_multiplier * self.LEADER_TARGET_IMPORTANCE * leader_reward) + followers_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.collisionOccurred():
            return True

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        leader_pos = states[0][0:3]
        leader_dist_to_target = np.linalg.norm(self.TARGET_POS - leader_pos)
        leader_reached_target = leader_dist_to_target < 0.0001
        
        followers_within_threshold = []
        for i in range(self.NUM_DRONES):
            follower_pos = states[i][0:3]
            follower_dist = np.linalg.norm(leader_pos - follower_pos)
            followers_within_threshold.append(follower_dist < self.FOLLOW_END_THRESHOLD)

        return leader_reached_target and all(followers_within_threshold)

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > self.FLYING_BOUND_RADIUS[0] or abs(states[i][1]) > self.FLYING_BOUND_RADIUS[1] or states[i][2] > self.FLYING_BOUND_RADIUS[2] # Truncate when a drones is too far away
                    or abs(states[i][7]) > .4 or abs(states[i][8]) > .4):  # Truncate when a drone is too tilted
                return True
            
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
    def collisionOccurred(self):
        for i in range(self.NUM_DRONES):
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[i], linkIndexA=-1, physicsClientId=self.CLIENT)
            if contact_points is not None and len(contact_points) > 0:
                return True
            
        return False
