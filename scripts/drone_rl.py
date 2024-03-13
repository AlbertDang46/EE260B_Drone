import rospy
import math
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, Point
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, ParamValue, State, WaypointList
from mavros_msgs.srv import CommandBool, ParamGet, ParamSet, SetMode, SetModeRequest, WaypointClear, WaypointPush
from pymavlink import mavutil
from sensor_msgs.msg import NavSatFix, Imu

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Drone_Control:
    """PX4 + Gazebo drone control class"""

    # Wait for drone services to come online and setup telemetry
    def __init__(self):
        # Drone telemetry
        self.altitude = Altitude()
        self.extended_state = ExtendedState()
        self.global_position = NavSatFix()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.local_velocity = TwistStamped()
        self.mission_wp = WaypointList()
        self.state = State()
        self.imu_data = Imu()
        self.mav_type = None

        # Ready states of drone telemetry
        self.sub_topics_ready = {
            key: False
            for key in [
                'alt', 'ext_state', 'global_pos', 'home_pos', 'local_pos', 
                'local_vel', 'mission_wp', 'state', 'imu'
            ]
        }

        # MAVROS services
        service_timeout = 30
        rospy.loginfo("Waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get', service_timeout)
            rospy.wait_for_service('mavros/param/set', service_timeout)
            rospy.wait_for_service('mavros/cmd/arming', service_timeout)
            rospy.wait_for_service('mavros/set_mode', service_timeout)
            rospy.wait_for_service('mavros/mission/push', service_timeout)
            rospy.wait_for_service('mavros/mission/clear', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            self.fail("failed to connect to services")

        # Methods to communicate with MAVROS services
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_param_srv = rospy.ServiceProxy('mavros/param/set', ParamSet)
        self.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)
        self.wp_push_srv = rospy.ServiceProxy('mavros/mission/push', WaypointPush)
        self.wp_clear_srv = rospy.ServiceProxy('mavros/mission/clear', WaypointClear)

        # MAVROS subscribers
        self.alt_sub = rospy.Subscriber('mavros/altitude', Altitude, self.altitude_callback)
        self.ext_state_sub = rospy.Subscriber('mavros/extended_state', ExtendedState, self.extended_state_callback)
        self.global_pos_sub = rospy.Subscriber('mavros/global_position/global', NavSatFix, self.global_position_callback)
        self.home_pos_sub = rospy.Subscriber('mavros/home_position/home', HomePosition, self.home_position_callback)
        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.local_position_callback)
        self.local_vel_sub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.local_velocity_callback)
        self.mission_wp_sub = rospy.Subscriber('mavros/mission/waypoints', WaypointList, self.mission_wp_callback)
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_callback)
        self.imu_data_sub = rospy.Subscriber('mavros/imu/data', Imu, self.imu_data_callback)

        # MAVROS publishers
        self.vel_setpoint_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)

    # Switch to offboard mode and arm the drone
    def startUp(self):
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)

    # Auto-land and de-arm drone
    def shutDown(self):
        self.set_mode('AUTO.LAND', 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 90, 0)
        self.set_arm(False, 5)
        self.tearDown()

    def tearDown(self):
        self.log_topic_vars()

    # Callback functions
    def altitude_callback(self, data):
        self.altitude = data

        # amsl has been observed to be nan while other fields are valid
        if not self.sub_topics_ready['alt'] and not math.isnan(data.amsl):
            self.sub_topics_ready['alt'] = True    

    def extended_state_callback(self, data):
        if self.extended_state.vtol_state != data.vtol_state:
            rospy.loginfo("VTOL state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE'][self.extended_state.vtol_state].name, 
                mavutil.mavlink.enums['MAV_VTOL_STATE'][data.vtol_state].name))

        if self.extended_state.landed_state != data.landed_state:
            rospy.loginfo("landed state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_LANDED_STATE'][self.extended_state.landed_state].name, 
                mavutil.mavlink.enums['MAV_LANDED_STATE'][data.landed_state].name))

        self.extended_state = data

        if not self.sub_topics_ready['ext_state']:
            self.sub_topics_ready['ext_state'] = True

    def global_position_callback(self, data):
        self.global_position = data

        if not self.sub_topics_ready['global_pos']:
            self.sub_topics_ready['global_pos'] = True

    def home_position_callback(self, data):
        self.home_position = data

        if not self.sub_topics_ready['home_pos']:
            self.sub_topics_ready['home_pos'] = True

    def local_position_callback(self, data):
        self.local_position = data

        if not self.sub_topics_ready['local_pos']:
            self.sub_topics_ready['local_pos'] = True

    def local_velocity_callback(self, data):
        self.local_velocity = data

        if not self.sub_topics_ready['local_vel']:
            self.sub_topics_ready['local_vel'] = True

    def mission_wp_callback(self, data):
        if self.mission_wp.current_seq != data.current_seq:
            rospy.loginfo("current mission waypoint sequence updated: {0}".format(data.current_seq))

        self.mission_wp = data

        if not self.sub_topics_ready['mission_wp']:
            self.sub_topics_ready['mission_wp'] = True

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][self.state.system_status].name, 
                mavutil.mavlink.enums['MAV_STATE'][data.system_status].name))

        self.state = data

        # mavros publishes a disconnected state message on init
        if not self.sub_topics_ready['state'] and data.connected:
            self.sub_topics_ready['state'] = True

    def imu_data_callback(self, data):
        self.imu_data = data

        if not self.sub_topics_ready['imu']:
            self.sub_topics_ready['imu'] = True

    # Helper methods to setup drone control
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in range(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(arm_set, 
            "failed to set arm | new arm: {0}, old arm: {1} | timeout(seconds): {2}".format(arm, old_arm, timeout))
        
    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in range(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(mode_set, 
            "failed to set mode | new mode: {0}, old mode: {1} | timeout(seconds): {2}".format(mode, old_mode, timeout))
        
    def set_param(self, param_id, param_value, timeout):
        """param: PX4 param string, ParamValue, timeout(int): seconds"""
        if param_value.integer != 0:
            value = param_value.integer
        else:
            value = param_value.real
        rospy.loginfo("setting PX4 parameter: {0} with value {1}".format(param_id, value))
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        for i in range(timeout * loop_freq):
            try:
                res = self.set_param_srv(param_id, param_value)
                if res.success:
                    rospy.loginfo("param {0} set to {1} | seconds: {2} of {3}".format(param_id, value, i / loop_freq, timeout))
                break
            except rospy.ServiceException as e:
                rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(res.success, 
            "failed to set param | param_id: {0}, param_value: {1} | timeout(seconds): {2}".format(param_id, value, timeout))

    def wait_for_topics(self, timeout):
        """wait for simulation to be ready, make sure we're getting topic info
        from all topics by checking dictionary of flag values set in callbacks,
        timeout(int): seconds"""
        rospy.loginfo("waiting for subscribed topics to be ready")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        simulation_ready = False
        for i in range(timeout * loop_freq):
            if all(value for value in self.sub_topics_ready.values()):
                simulation_ready = True
                rospy.loginfo("simulation topics ready | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(simulation_ready, 
            "failed to hear from all subscribed simulation topics | topic ready flags: {0} | timeout(seconds): {1}".format(
                self.sub_topics_ready, 
                timeout))
        
    def wait_for_landed_state(self, desired_landed_state, timeout, index):
        rospy.loginfo("waiting for landed state | state: {0}, index: {1}".format(
            mavutil.mavlink.enums['MAV_LANDED_STATE'][desired_landed_state].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        landed_state_confirmed = False
        for i in range(timeout * loop_freq):
            if self.extended_state.landed_state == desired_landed_state:
                landed_state_confirmed = True
                rospy.loginfo("landed state confirmed | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(landed_state_confirmed, 
            "landed state not detected | desired: {0}, current: {1} | index: {2}, timeout(seconds): {3}".format(
                mavutil.mavlink.enums['MAV_LANDED_STATE'][desired_landed_state].name, 
                mavutil.mavlink.enums['MAV_LANDED_STATE'][self.extended_state.landed_state].name,
                index, 
                timeout))
        
    def wait_for_vtol_state(self, transition, timeout, index):
        """Wait for VTOL transition, timeout(int): seconds"""
        rospy.loginfo("waiting for VTOL transition | transition: {0}, index: {1}".format(
            mavutil.mavlink.enums['MAV_VTOL_STATE'][transition].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        transitioned = False
        for i in range(timeout * loop_freq):
            if self.extended_state.vtol_state == transition:
                rospy.loginfo("transitioned | seconds: {0} of {1}".format(i / loop_freq, timeout))
                transitioned = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(transitioned,
            "transition not detected | desired: {0}, current: {1} | index: {2} timeout(seconds): {3}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE'][transition].name,
                mavutil.mavlink.enums['MAV_VTOL_STATE'][self.extended_state.vtol_state].name, 
                index, 
                timeout))
        
    def clear_wps(self, timeout):
        """timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        wps_cleared = False
        for i in range(timeout * loop_freq):
            if not self.mission_wp.waypoints:
                wps_cleared = True
                rospy.loginfo("clear waypoints success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.wp_clear_srv()
                    if not res.success:
                        rospy.logerr("failed to send waypoint clear command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(wps_cleared, "failed to clear waypoints | timeout(seconds): {0}".format(timeout))

    def send_wps(self, waypoints, timeout):
        """waypoints, timeout(int): seconds"""
        rospy.loginfo("sending mission waypoints")
        if self.mission_wp.waypoints:
            rospy.loginfo("FCU already has mission waypoints")

        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        wps_sent = False
        wps_verified = False
        for i in range(timeout * loop_freq):
            if not wps_sent:
                try:
                    res = self.wp_push_srv(start_index=0, waypoints=waypoints)
                    wps_sent = res.success
                    if wps_sent:
                        rospy.loginfo("waypoints successfully transferred")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            else:
                if len(self.mission_wp.waypoints) == len(waypoints):
                    rospy.loginfo("number of waypoints transferred: {0}".format(len(waypoints)))
                    wps_verified = True

            if wps_sent and wps_verified:
                rospy.loginfo("send waypoints success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(wps_sent and wps_verified, 
            "mission could not be transferred and verified | timeout(seconds): {0}".format(timeout))
        
    def wait_for_mav_type(self, timeout):
        """Wait for MAV_TYPE parameter, timeout(int): seconds"""
        rospy.loginfo("waiting for MAV_TYPE")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        res = False
        for i in range(timeout * loop_freq):
            try:
                res = self.get_param_srv('MAV_TYPE')
                if res.success:
                    self.mav_type = res.value.integer
                    rospy.loginfo("MAV_TYPE received | type: {0} | seconds: {1} of {2}".format(
                        mavutil.mavlink.enums['MAV_TYPE'][self.mav_type].name, 
                        i / loop_freq, timeout))
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(res.success, "MAV_TYPE param get failed | timeout(seconds): {0}".format(timeout))

    def log_topic_vars(self):
        """log the state of topic variables"""
        rospy.loginfo("========================")
        rospy.loginfo("===== topic values =====")
        rospy.loginfo("========================")
        rospy.loginfo("altitude:\n{}".format(self.altitude))
        rospy.loginfo("========================")
        rospy.loginfo("extended_state:\n{}".format(self.extended_state))
        rospy.loginfo("========================")
        rospy.loginfo("global_position:\n{}".format(self.global_position))
        rospy.loginfo("========================")
        rospy.loginfo("home_position:\n{}".format(self.home_position))
        rospy.loginfo("========================")
        rospy.loginfo("local_position:\n{}".format(self.local_position))
        rospy.loginfo("========================")
        rospy.loginfo("mission_wp:\n{}".format(self.mission_wp))
        rospy.loginfo("========================")
        rospy.loginfo("state:\n{}".format(self.state))
        rospy.loginfo("========================")


class Drone_Environment:
    """Reinforcement learning for PX4 drone with ROS/Gazebo"""

    # Initialize reinforcement learning parameters
    def __init__(self, goal_pos=[50.0, 60.0, 5.0]):
        self.drone_control = Drone_Control()
        self.drone_control.startUp()

        self.hertz = 10
        self.rate = rospy.Rate(self.hertz)

        self.timestep = 0
        self.time_penalty_factor = 1.05  # Reward penalty increases the more time has passed
        self.max_timesteps = 200  # Scenario is truncated if too many timesteps have passed

        self.goal_pos = goal_pos
        self.goal_threshold = 0.2   # Scenario successfully terminates if drone reaches the goal within this threshold
        self.prev_dist_to_goal = 0.0
        
    # Set drone velocity to 0 and reset drone position
    def reset(self):
        zero_vel = Twist()
        zero_vel.linear.x = 0
        zero_vel.linear.y = 0
        zero_vel.linear.z = 0
        zero_vel.angular.x = 0
        zero_vel.angular.y = 0
        zero_vel.angular.z = 0
        
        self.drone_control.vel_setpoint_pub.publish(zero_vel)
        self.rate.sleep()
        # TODO: Use Gazebo command to reset drone position
        
        local_pos = self.drone_control.local_position.pose.point
        local_vel = self.drone_control.local_velocity.twist.linear
        reset_state = np.array([local_pos.x, local_pos.y, local_pos.z, local_vel.x, local_vel.y, local_vel.z])

        return reset_state

    def step(self, velocity_action):
        vel = Twist()
        vel.linear.x = velocity_action[0]
        vel.linear.y = velocity_action[1]
        vel.linear.z = velocity_action[2]
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0

        self.drone_control.vel_setpoint_pub.publish(vel)
        self.rate.sleep()
        self.timestep += 1
        
        local_pos = self.drone_control.local_position.pose.point
        local_vel = self.drone_control.local_velocity.twist.linear
        next_state = np.array([local_pos.x, local_pos.y, local_pos.z, local_vel.x, local_vel.y, local_vel.z])

        dist_to_goal = math.dist([local_pos.x, local_pos.y, local_pos.z], self.goal_pos)
        reward = self.prev_dist_to_goal - dist_to_goal - ((1.0 / self.hertz) * self.time_penalty_factor**self.timestep)
        
        terminated = dist_to_goal < self.goal_threshold  # TODO: also check for collisions with obstacles or floor
        truncated = self.timestep >= self.max_timesteps

        return (next_state, reward, terminated, truncated)
        
    # Shutdown drone and reset drone position
    def shutdown(self):
        self.drone_control.shutDown()
        # TODO: Use Gazebo command to reset drone position
       

class Drone_Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
        of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, input: torch.Tensor):
        """Conditioned on the observation, returns the mean and standard deviation
        of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(input.float())

        action_means = self.policy_mean_net(shared_features)
        
        # Keep standard deviation positive even if stddev_net outputs a negative value
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


class Drone_REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Drone_Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray):
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        # Get predicted mean and standard deviation from policy network
        state_tensor = torch.tensor(state)
        action_means, action_stddevs = self.net(state_tensor)

        # Create a normal distribution from the predicted mean and standard deviation and sample an action
        norm_distrib = Normal(action_means + self.eps, action_stddevs + self.eps)
        action = norm_distrib.sample()

        prob = norm_distrib.log_prob(action)
        self.probs.append(prob)

        action = action.tolist()
        return action

    def update(self):
        """Updates the policy network's weights."""
        running_G = 0
        Gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_G = R + self.gamma * running_G
            Gs.insert(0, running_G)

        Gs_tensor = torch.tensor(Gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for G, log_prob in zip(Gs_tensor, self.probs):
            loss += (-1) * G * log_prob.mean()

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


def print_neural_net_parameters(neural_net):
    for param in neural_net.parameters():
        print(type(param), param.size(), param)


rospy.init_node("drone_rl_py")

drone_env = Drone_Environment()
total_num_episodes = 5000

obs_space_dims = 6  # Drone State: x,y,z local coordinates and x,y,z linear velocities
action_space_dims = 3   # Drone Action: x,y,z linear velocities
drone_agent = Drone_REINFORCE(obs_space_dims, action_space_dims)

for episode in range(total_num_episodes):
    obs = drone_env.reset()

    done = False
    while not done:
        action = drone_agent.sample_action(obs)
        obs, reward, terminated, truncated = drone_env.step(action)
        drone_agent.rewards.append(reward)
        done = terminated or truncated
    
    drone_agent.update()
    print_neural_net_parameters("Shared Net Parameters: " + str(drone_agent.net.shared_net))
    print_neural_net_parameters("Means Net Parameters: " + str(drone_agent.net.policy_mean_net))
    print_neural_net_parameters("StdDevs Net Parameters: " + str(drone_agent.net.policy_stddev_net))

drone_env.shutdown()