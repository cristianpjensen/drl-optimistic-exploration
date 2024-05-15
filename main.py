import os
import random
import shutil
import string
from glob import glob
from typing import Tuple

import gymnasium as gym
import neptune
import numpy as np
import torch
from dotenv import load_dotenv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.time_limit import TimeLimit
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment
from tqdm import tqdm

from agent import AGENTS
from agent.agent import Agent

load_dotenv()


ex = Experiment()

# Add Neptune observer for storing logs
run = neptune.init_run(
    project=os.environ["NEPTUNE_PROJECT"],
    api_token=os.environ["NEPTUNE_API_TOKEN"],
    source_files=["**/*.py"],
)
ex.observers.append(NeptuneObserver(run))


@ex.config
def config():
    env_name = "ALE/SpaceInvaders-v5"
    num_envs = 32
    env_config = {}

    # Agent identifier
    agent_id = "atari-dqn"

    # Replay buffer sizes
    batch_size = 256
    min_buffer_size = 50_000
    max_buffer_size = 1_000_000

    # How many steps to train the agent for
    # 50M steps is 200M frames when accounting for frame skipping
    train_steps = 50_000_000
    
    # How many episodes to test on after training
    test_episodes = 10

    # How many times to intermittently test the model between training steps
    train_test_interval = 500_000

    # Discount factor (https://www.nature.com/articles/nature14236)
    gamma = 0.99


@ex.main
def main(
    env_name: str,
    num_envs: int,
    env_config: dict,
    agent_id: str,
    batch_size: int,
    min_buffer_size: int,
    max_buffer_size: int,
    train_steps: int,
    test_episodes: int,
    train_test_interval: int,
    gamma: float,
):
    NAME = get_random_id()
    os.mkdir(NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Run ID:", NAME)
    print("DO NOT DELETE THIS DIRECTORY!")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name())

    if "ALE" in env_name:
        envs = gym.make_vec(
            env_name,
            **env_config,
            num_envs=num_envs, 
            wrappers=[
                # The ALE environments already have frame skipping
                lambda env: AtariPreprocessing(env, frame_skip=1, screen_size=84),
                lambda env: FrameStack(env, 4),
                lambda env: TimeLimit(env, 5 * 30 * 60),
            ],
        )
        test_env = gym.make(env_name, render_mode="rgb_array", **env_config)
        test_env = AtariPreprocessing(test_env, frame_skip=1, screen_size=84)
        test_env = FrameStack(test_env, 4)
        test_env = TimeLimit(test_env, 5 * 30 * 60)

        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            min_buffer_size=min_buffer_size,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            device=device,
        )
    else:
        envs = gym.make_vec(
            env_name,
            **env_config,
            num_envs=num_envs,
            wrappers=[FlattenObservation, lambda env: FrameStack(env, 1)]
        )
        test_env = gym.make(env_name, render_mode="rgb_array", **env_config)
        test_env = FlattenObservation(test_env)
        test_env = FrameStack(test_env, 1)

        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            min_buffer_size=min_buffer_size,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            device=device,
        )

    test_env = RecordVideo(test_env, f"{NAME}/videos", disable_logger=True, episode_trigger=lambda _: True)

    config = {
        "gamma": gamma,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "train_start":  min_buffer_size,
    }
    env_tmp = gym.make(env_name, **env_config)
    if hasattr(env_tmp.observation_space, "n"):
        config["n_states"] = env_tmp.observation_space.n

    if hasattr(env_tmp.action_space, "n"):
        config["n_actions"] = env_tmp.action_space.n

    agent.setup(config)
    envs.metadata["render_fps"] = 30

    train_agent(envs, test_env, agent, gamma, train_steps, train_test_interval, max(batch_size // 8, 1), run, name=NAME)
    envs.close()

    test_ep_return, test_discounted_ep_return, test_timesteps = test_agent(test_env, agent, gamma, test_episodes)
    for i in range(test_episodes):
        run["test/discounted_return"].append(test_discounted_ep_return[i])
        run["test/undiscounted_return"].append(test_ep_return[i])
        run["test/time_alive"].append(test_timesteps[i])

    test_env.close()

    for i, video in enumerate(sorted(glob(f"{NAME}/videos/*.mp4"))):
        run[f"test/episode-{i}.mp4"].upload(video, wait=True)

    # Save weights
    saved = agent.save(f"{NAME}/weights")
    if saved:
        shutil.make_archive(f"{NAME}/weights", "zip", f"{NAME}/weights")
        run["weights"].upload(f"{NAME}/weights.zip", wait=True)

    # Clean up
    shutil.rmtree(NAME)


def train_agent(
    envs: gym.vector.VectorEnv,
    test_env: gym.Env,
    agent: Agent,
    gamma: float,
    total_steps: int,
    test_interval: int,
    update_freq: int,
    run: neptune.Run,
    name: str = ""
):
    timesteps_trained = 0
    state, _ = envs.reset()

    ep_timesteps = np.zeros(envs.num_envs)
    ep_return = np.zeros(envs.num_envs)
    discounted_ep_return = np.zeros(envs.num_envs)

    episode_histories = [[] for _ in range(envs.num_envs)]

    with tqdm(total=total_steps, desc="Training", miniters=total_steps // 10_000) as pbar:
        while timesteps_trained < total_steps:
            # Sample action and step in the environment
            action = agent.act(state, train=True)
            next_state, reward, terminated, truncated, _ = envs.step(action)

            # Update statistics
            ep_return += reward
            discounted_ep_return += (gamma ** ep_timesteps) * reward
            ep_timesteps += 1

            done = terminated | truncated

            for i in range(envs.num_envs):
                timesteps_trained += 1

                # Update episode history
                episode_histories[i].append((state[i], action[i], reward[i], terminated[i]))

                # Update policy
                if agent.replay_buffer.is_ready() and timesteps_trained % update_freq == 0:
                    agent.update_policy()

                # Intermittent testing
                if timesteps_trained % test_interval == 0:
                    # Do 2 test episodes, since 1 is not enough to get a good estimate (Machado et al. 2018, Section 4.1.2)
                    test_ep_return, test_discounted_ep_return, test_timesteps = test_agent(test_env, agent, gamma, 2)
                    run["train/intermittent-testing/test_undiscounted_return"].append(step=timesteps_trained, value=test_ep_return.mean())
                    run["train/intermittent-testing/test_discounted_return"].append(step=timesteps_trained, value=test_discounted_ep_return.mean())
                    run["train/intermittent-testing/steps_alive"].append(step=timesteps_trained, value=test_timesteps.mean())

                    # Upload video to Neptune and delete it
                    for video in glob(f"{name}/videos/*.mp4"):
                        run[f"train/intermittent-testing/videos/training-steps-{timesteps_trained}.mp4"].upload(video, wait=True)
                        os.remove(video)

                # Report statistics of finished episodes and add to replay buffer
                if done[i]:
                    run["train/undiscounted_return"].append(step=timesteps_trained, value=ep_return[i])
                    run["train/discounted_return"].append(step=timesteps_trained, value=discounted_ep_return[i])

                    agent.replay_buffer.push_episode(episode_histories[i])
                    episode_histories[i] = []

            # Reset done environments
            ep_timesteps[done] = 0
            ep_return[done] = 0
            discounted_ep_return[done] = 0

            # Log anything that is relevant to the agent
            agent.log(run)

            # Update state
            state = next_state
            pbar.update(envs.num_envs)

    envs.reset()


def test_agent(
    env: gym.Env,
    agent: Agent,
    gamma: float,
    episodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ep_return = np.zeros(episodes)
    discounted_ep_return = np.zeros(episodes)
    timesteps = np.zeros(episodes)

    for i in range(episodes):
        done = False
        state, _ = env.reset()

        while not done:
            # Make sure the state has the same form as in the vectorized environment
            s = np.expand_dims(np.array(state), 0)
            action = agent.act(s, train=False)[0]
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update statistics
            ep_return[i] += reward
            discounted_ep_return[i] += (gamma ** timesteps[i]) * reward
            timesteps[i] += 1

            # Check if done and update state
            done = terminated or truncated
            state = next_state

    return ep_return, discounted_ep_return, timesteps


def get_random_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


if __name__ == "__main__":
    ex.run_commandline()
