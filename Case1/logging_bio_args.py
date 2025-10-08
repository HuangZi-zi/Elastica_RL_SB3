__doc__ = """This script is to train or run a policy for the arm following randomly moving target. 
Case 1 in CoRL 2020 paper."""
import os
import numpy as np
import sys

import argparse
import matplotlib
import matplotlib.pyplot as plt

# Import stable baseline
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ddpg.policies import MlpPolicy as MlpPolicy_DDPG
from stable_baselines3.td3.policies import MlpPolicy as MlpPolicy_TD3
from stable_baselines3.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines3 import DDPG, PPO, TD3, SAC

# Import simulation environment
from set_environment import Environment


def get_valid_filename(s):
    import re

    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(title + ".png")
    plt.close()


if __name__ == "__main__":
    # If True, train. Otherwise, run trained policy
    TRAIN = True

    total_timesteps = 1e6
    SEED = 42
    timesteps_per_batch = 8000
    algo_name = "PPO"

    # Mode 4 corresponds to randomly moving target
    mode = 4

    # Set simulation final time
    final_time = 10
    # Number of control points
    number_of_control_points = 6
    # target position
    target_position = np.array([-0.4, 0.6, 0.2])
    # learning step skip
    num_steps_per_update = 7
    # alpha and beta spline scaling factors in normal/binormal and tangent directions respectively
    alpha = 75
    beta = 75

    sim_dt = 1.0e-4


    if algo_name == "DDPG":
        MLP = MlpPolicy_DDPG
        algo = DDPG
        batchsize = "nb_rollout_steps"
        offpolicy = True
    elif algo_name == "TD3":
        MLP = MlpPolicy_TD3
        algo = TD3
        batchsize = "train_freq"
        offpolicy = True
    elif algo_name == "SAC":
        MLP = MlpPolicy_SAC
        algo = SAC
        batchsize = "train_freq"
        offpolicy = True
    else:
        MLP = MlpPolicy
        algo = PPO
        batchsize = "n_steps"
        offpolicy = False



    max_rate_of_change_of_activation = np.inf
    print("rate of change", max_rate_of_change_of_activation)



    env = Environment(
        final_time=final_time,
        num_steps_per_update=num_steps_per_update,
        number_of_control_points=number_of_control_points,
        alpha=alpha,
        beta=beta,
        COLLECT_DATA_FOR_POSTPROCESSING=not TRAIN,
        mode=mode,
        target_position=target_position,
        target_v=0.5,
        boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
        E=1e7,
        sim_dt=sim_dt,
        n_elem=20,
        UDC=0.01,
        num_obstacles=0,
        dim=3.0,
        max_rate_of_change_of_activation=max_rate_of_change_of_activation,
    )

    from stable_baselines3.common.env_checker import check_env
    # It will check your custom environment and output additional warnings if needed

    from stable_baselines3.common.env_util import make_vec_env
    # new_env = make_vec_env(env, n_envs=4)
    # check_env(env)
    # vec_env = make_vec_env(env, n_envs=4)

    name = str(algo_name) + "_3d-tracking_id"
    identifer = name + "-" + str(timesteps_per_batch) + "_" + str(SEED)


    if TRAIN:
        log_dir = "./log_" + identifer + "/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)


    from stable_baselines3.common.results_plotter import ts2xy, plot_results
    from stable_baselines3.common import results_plotter

    if TRAIN:
        if offpolicy:
            if algo_name == "TD3":
                items = {
                    "policy": MLP,
                    "buffer_size": int(timesteps_per_batch),
                    "learning_starts": int(50e3),
                }
            else:
                items = {"policy": MLP, "buffer_size": int(timesteps_per_batch)}
        else:
            items = {
                "policy": MLP,
                batchsize: timesteps_per_batch,
            }

        model = algo(env=env, verbose=1, seed=SEED, **items)
        model.set_env(env)

        model.learn(total_timesteps=int(total_timesteps))
        # library helper
        plot_results(
            [log_dir],
            int(total_timesteps),
            results_plotter.X_TIMESTEPS,
            "PPO muscle" + identifer,
        )
        plt.savefig("convergence_plot" + identifer + ".png")
        model.save("policy-" + identifer)

    else:
        # Use trained policy for the simulation.
        model = PPO.load("policy-" + identifer)

        obs, _ = env.reset()

        done = False
        score = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, _, info = env.step(action)
            score += rewards
            if info["ctime"] > final_time:
                break
        print("Final Score:", score)
        env.post_processing(
            filename_video="video-" + identifer + ".mp4", SAVE_DATA=True,
        )
