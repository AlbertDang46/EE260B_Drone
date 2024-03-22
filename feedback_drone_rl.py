"""
Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.
"""

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from VelocityRLAviary import VelocityRLAviary


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType.RGB  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType.VEL  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, record_video=DEFAULT_RECORD_VIDEO):

    #### Directory where model is saved
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    #### Create training and evaluation environments
    train_env = make_vec_env(VelocityRLAviary,
                             env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                             n_envs=1,
                             seed=0)
    
    eval_env = VelocityRLAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's action/observation spaces
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the PPO model -- model takes in both RGBA camera input and drone kinematics input
    model = PPO('MultiInputPolicy', train_env, verbose=1)

    #### Target cumulative rewards (problem-dependent)
    target_reward = 200000
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    
    model.learn(total_timesteps=int(1e5),
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename + '/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename + '/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    #### Pause between end of training and best model evaluation
    input("Press Enter to continue...")

    #### Load best model
    if os.path.isfile(filename + '/best_model.zip'):
        path = filename + '/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)

    model = PPO.load(path)

    #### Create one env to visualize and one env to quickly get an evaluation
    test_env = VelocityRLAviary(gui=gui,
                                num_drones=DEFAULT_AGENTS,
                                obs=DEFAULT_OBS,
                                act=DEFAULT_ACT,
                                record=record_video)
    
    test_env_nogui = VelocityRLAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=DEFAULT_AGENTS, output_folder=output_folder)

    #### Evaluate model
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    #### Show (and record a video of) the model's performance
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs_kin = obs["kinematics"].squeeze()
        act_vel = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        
        for d in range(DEFAULT_AGENTS):
            drone_state = np.hstack([obs_kin[d][0:3], np.zeros(4), obs_kin[d][3:12], act_vel[d]])
            logger.log(drone=d,
                       timestamp=i / test_env.CTRL_FREQ,
                       state=drone_state,
                       control=np.zeros(12))

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)

        if terminated:
            obs = test_env.reset(seed=42, options={})

    test_env.close()

    #### Plot drone kinematic data over course of episode
    if plot:
        logger.plot()


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
