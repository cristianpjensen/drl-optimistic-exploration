import os
import shutil
from typing import Tuple

import gymnasium as gym
import neptune
import torch
import numpy as np
from dotenv import load_dotenv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.flatten_observation import FlattenObservation
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
    num_envs = 4
    env_config = {}

    # Agent identifier
    agent_id = "atari-dqn"

    # Replay buffer sizes
    batch_size = 32
    min_buffer_size = 100
    max_buffer_size = 1_000_000

    # How many steps to train the agent for
    train_steps = 50_000_000
    
    # How many episodes to test on after training
    test_episodes = 100

    # How many times to intermittently test the model between training steps
    train_test_interval = 5000

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if "ALE" in env_name:
        envs = gym.make_vec(
            env_name,
            **env_config,
            num_envs=num_envs, 
            wrappers=[
                # The ALE environments already have frame skipping
                lambda env: AtariPreprocessing(env, frame_skip=1, screen_size=84),
                lambda env: FrameStack(env, 4),
            ],
        )
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
        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            min_buffer_size=min_buffer_size,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            device=device,
        )

    config = { "gamma": gamma, "train_steps": train_steps }
    if hasattr(envs.single_observation_space, "n"):
        config["n_states"] = envs.single_observation_space.n

    if hasattr(envs.single_action_space, "n"):
        config["n_actions"] = envs.single_action_space.n

    agent.setup(config)
    envs.metadata["render_fps"] = 30

    train_agent(envs, agent, gamma, train_steps, train_test_interval, run)

    # Add video recorder wrapper only on the first test episode
    envs.reset()

    if "ALE" in env_name:
        envs = gym.make_vec(
            env_name,
            **env_config,
            render_mode="rgb_array", 
            num_envs=num_envs, 
            wrappers=[
                # The ALE environments already have frame skipping
                lambda env: AtariPreprocessing(env, frame_skip=1, screen_size=84),
                lambda env: FrameStack(env, 4),
                lambda env: RecordVideo(env, "videos", name_prefix="test", disable_logger=True, episode_trigger=lambda t: t == 0),
            ],
        )
    else:
        envs = gym.make_vec(
            env_name,
            **env_config,
            render_mode="rgb_array",
            num_envs=num_envs,
            wrappers=[
                FlattenObservation,
                lambda env: FrameStack(env, 1),
                lambda env: RecordVideo(env, "videos", name_prefix="test", disable_logger=True, episode_trigger=lambda t: t == 0),
            ],
        )

    test_ep_return, test_discounted_ep_return, test_timesteps = test_agent(envs, agent, gamma, test_episodes)
    for i in range(test_episodes):
        run["test/discounted_return"].append(test_discounted_ep_return[i])
        run["test/undiscounted_return"].append(test_ep_return[i])
        run["test/time_alive"].append(test_timesteps[i])

    envs.close()

    run["test/video"].upload(f"videos/test-episode-0.mp4", wait=True)
    shutil.rmtree("videos")

    # Save weights and clean up
    saved = agent.save("weights")
    if saved:
        shutil.make_archive("weights", "zip", "weights")
        run["weights"].upload("weights.zip", wait=True)
        shutil.rmtree("weights")
        os.remove("weights.zip")


def train_agent(
    envs: gym.vector.VectorEnv,
    agent: Agent,
    gamma: float,
    total_steps: int,
    test_interval: int,
    run: neptune.Run,
):
    timesteps_trained = 0
    state, _ = envs.reset()

    ep_timesteps = np.zeros(envs.num_envs)
    ep_return = np.zeros(envs.num_envs)
    discounted_ep_return = np.zeros(envs.num_envs)

    episode_histories = [[] for _ in range(envs.num_envs)]

    with tqdm(total=total_steps, desc="Training") as pbar:
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
                episode_histories[i].append((state[i], action[i], reward[i], next_state[i], done[i]))

                # Update policy
                if agent.replay_buffer.is_ready() and timesteps_trained % 4 == 0:
                    agent.update_policy()

                # Intermittent testing
                if timesteps_trained % test_interval == 0:
                    test_ep_return, test_discounted_ep_return, test_timesteps = test_agent(envs, agent, gamma, 1)
                    run["train/test_undiscounted_return"].append(step=timesteps_trained, value=test_ep_return[0])
                    run["train/test_discounted_return"].append(step=timesteps_trained, value=test_discounted_ep_return[0])
                    run["train/test_steps_alive"].append(step=timesteps_trained, value=test_timesteps[0])

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
    envs: gym.vector.VectorEnv,
    agent: Agent,
    gamma: float,
    episodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Takes a vectorized environment, but only tests on the first environment for `episodes`
    episodes."""

    ep_return = np.zeros(episodes)
    discounted_ep_return = np.zeros(episodes)
    timesteps = np.zeros(episodes)

    for i in range(episodes):
        done = False
        state, _ = envs.reset()

        while not done:
            # Sample action and step in the environment
            action = agent.act(state, train=False)
            next_state, reward, terminated, truncated, _ = envs.step(action)

            # Update statistics
            ep_return[i] += reward[0]
            discounted_ep_return[i] += (gamma ** timesteps[i]) * reward[0]
            timesteps[i] += 1

            # Check if done and update state
            done = terminated[0] or truncated[0]
            state = next_state

    return ep_return, discounted_ep_return, timesteps


if __name__ == "__main__":
    ex.run_commandline()
