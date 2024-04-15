import os
import shutil
from collections import deque
from typing import Tuple

import gymnasium as gym
import neptune
import numpy as np
import torch
from dotenv import load_dotenv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.normalize import NormalizeReward
from gymnasium.wrappers.record_video import RecordVideo
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
ex.observers.append(NeptuneObserver(run=run))


@ex.config
def config():
    env_name = "ALE/SpaceInvaders-v5"
    # By using 4 environments, it is the same as updating every 4th timestep, as in
    # (https://www.nature.com/articles/nature14236). Make sure if you use more environments, you
    # update the batch size accordingly (e.g. if num_envs=8, batch_size=64).
    num_envs = 4
    batch_size = 32
    agent_id = "atari-dqn"
    train_steps = 10000000
    test_episodes = 5

    # Every 10 training episodes, run a test episode
    train_test_interval = 10

    # Default: 0.99 (https://www.nature.com/articles/nature14236).
    gamma = 0.99

    agent_config = {
        "gamma": gamma,
    }


@ex.main
def main(
    env_name: str,
    num_envs: int,
    batch_size: int,
    agent_id: str,
    agent_config: dict,
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
            render_mode="rgb_array", 
            num_envs=num_envs, 
            wrappers=[
                # The ALE environments already have frame skipping
                lambda env: AtariPreprocessing(env, frame_skip=1, screen_size=84),
                NormalizeReward,
            ],
        )
        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            batch_size=batch_size,
            frame_stack=4,
            device=device,
        )
    else:
        envs = gym.vector.make(
            env_name,
            render_mode="rgb_array",
            num_envs=num_envs,
            wrappers=[NormalizeReward]
        )
        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            batch_size=batch_size,
            frame_stack=1,
            device=device,
        )

    envs.metadata["render_fps"] = 30

    agent.setup(agent_config)

    timesteps_trained = 0
    episodes_trained = 0

    with tqdm(total=train_steps, desc="Training") as pbar:
        while timesteps_trained < train_steps:
            discounted_return, undiscounted_return, timesteps = run_episode(envs, agent, gamma, train=True, log=True)
            timesteps_trained += timesteps

            if episodes_trained % train_test_interval == 0:
                run["train/test_discounted_return"].append(step=timesteps_trained, value=discounted_return.mean())
                run["train/test_undiscounted_return"].append(step=timesteps_trained, value=undiscounted_return.mean())

            run["train/discounted_return"].append(step=timesteps_trained, value=discounted_return.mean())
            run["train/undiscounted_return"].append(step=timesteps_trained, value=undiscounted_return.mean())
            run["train/replay_buffer_size"].append(step=timesteps_trained, value=len(agent.replay_buffer))

            pbar.update(timesteps)
            episodes_trained += 1

    # Add video recorder wrapper only on the first test episode
    envs.reset()
    envs = RecordVideo(
        envs,
        "videos",
        name_prefix="test",
        disable_logger=True,
        episode_trigger=lambda t: t == 0,
    )

    for ep in tqdm(range(test_episodes), desc="Testing"):
        discounted_return, undiscounted_return, _ = run_episode(envs, agent, gamma, train=False, log=False)
        for i in range(envs.num_envs):
            run["test/discounted_return"].append(discounted_return[i])
            run["test/undiscounted_return"].append(undiscounted_return[i])

        if ep == 0:
            run["test/video"].upload(f"videos/test-episode-{ep}.mp4", wait=True)

    envs.close()

    saved = agent.save("weights")
    if saved:
        shutil.make_archive("weights", "zip", "weights")
        run["weights"].upload("weights.zip")
        shutil.rmtree("weights")
        os.remove("weights.zip")

    # Cleanup
    shutil.rmtree("videos")


def run_episode(env: gym.Env, agent: Agent, gamma: float, train=True, log=False) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run an episode in the environment with the given agent.

    Args:
        env: Environment.
        agent: Agent.
        gamma: Discount factor.
        train: If enabled, the transitions are added to the replay buffer, and
            the agent's policy is updated each iteration.
        log: If enabled, call the log method of the agent, which logs anything
            relevant to the agent.

    Returns:
        - The discounted return of the episode.
        - The undiscounted return of the episode.
        - Number of steps

    """

    state, _ = env.reset()
    ep_return = 0
    undiscounted_ep_return = 0
    t = 0

    # Make sure the observations given to the agent are frame stacked as the replay buffer samples
    frame_stack = agent.replay_buffer.frame_stack
    frames = deque(maxlen=frame_stack)

    # Fill up frame stack with zeros
    for _ in range(frame_stack):
        frames.append(np.zeros_like(state))

    done = np.array([False] * env.num_envs)
    while not np.all(done):
        frames.append(state)

        # Set batch size first
        observation = np.array(frames)
        observation = observation.swapaxes(0, 1)

        action = agent.act(observation, train)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if train:
            for i in range(env.num_envs):
                if not done[i]:
                    agent.replay_buffer.push(state[i], action[i], reward[i], terminated[i])

            # Update frequency 4: https://www.nature.com/articles/nature14236
            if agent.replay_buffer.is_ready():
                agent.update_policy()

        if log:
            agent.log(run)

        ep_return += (gamma**t) * reward
        undiscounted_ep_return += reward
        t += int(np.logical_not(done).sum())

        state = next_state
        done = done | terminated | truncated

    return ep_return, undiscounted_ep_return, t


if __name__ == "__main__":
    ex.run_commandline()
