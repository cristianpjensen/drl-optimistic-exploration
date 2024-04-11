import os
import shutil

import gymnasium as gym
import neptune
import torch
import numpy as np
from collections import deque
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
    env_name = "LunarLander-v2"
    agent_id = "random"
    train_steps = 1000000
    test_episodes = 5

    # Every 10 training episodes, run a test episode
    train_test_interval = 10
    gamma = 0.99

    agent_config = {
        "gamma": gamma,
    }


@ex.main
def main(
    env_name: str,
    agent_id: str,
    agent_config: dict,
    train_steps: int,
    test_episodes: int,
    train_test_interval: int,
    gamma: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = gym.make(env_name, render_mode="rgb_array")
    env = NormalizeReward(env, gamma=gamma)

    # Preprocessing for Atari games. Make sure to only use the ALE namespace, such that this
    # condition does not fail.
    if env.spec.namespace == "ALE":
        # The ALE environments already have frame skipping
        env = AtariPreprocessing(env, frame_skip=1, screen_size=84)
        agent = AGENTS[agent_id](env.observation_space, env.action_space, frame_stack=4, device=device)
    else:
        agent = AGENTS[agent_id](env.observation_space, env.action_space, frame_stack=1, device=device)

    agent.setup(agent_config)

    timesteps_trained = 0
    episodes_trained = 0

    with tqdm(total=train_steps, desc="Training") as pbar:
        while timesteps_trained < train_steps:
            if episodes_trained % train_test_interval == 0:
                discounted_return, undiscounted_return, _ = run_episode(env, agent, gamma, train=False, log=False)
                run["train/test_discounted_return"].append(discounted_return)
                run["train/test_undiscounted_return"].append(undiscounted_return)

            discounted_return, undiscounted_return, timesteps = run_episode(env, agent, gamma, train=True, log=False)
            timesteps_trained += timesteps

            run["train/timesteps"].append(timesteps_trained)
            run["train/discounted_return"].append(discounted_return)
            run["train/undiscounted_return"].append(undiscounted_return)

            pbar.update(timesteps)

            episodes_trained += 1

    # Add video recorder wrapper only on the first test episode
    env.reset()
    env = RecordVideo(
        env,
        "videos",
        name_prefix="test",
        disable_logger=True,
        episode_trigger=lambda t: t == 0,
    )

    for ep in tqdm(range(test_episodes), desc="Testing"):
        discounted_return, undiscounted_return, _ = run_episode(env, agent, gamma, train=False, log=True)
        run["test/discounted_return"].append(discounted_return)
        run["test/undiscounted_return"].append(undiscounted_return)

        if ep == 0:
            run["test/video"].upload(f"videos/test-episode-{ep}.mp4", wait=True)

    env.close()

    # Remove videos
    shutil.rmtree("videos")


# TODO: Mark loss of life as terminal state, but don't reset the environment. Supposedly, this
# leads to more efficient training, according to https://github.com/jacobaustin123/pytorch-dqn.


def run_episode(env: gym.Env, agent: Agent, gamma: float, train=True, log=False):
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

    done = False
    while not done:
        frames.append(state)
        observation = np.array(frames)

        action = agent.act(observation, train)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if train:
            agent.replay_buffer.push(state, action, reward, terminal=terminated or truncated)

            if agent.replay_buffer.is_ready() and t % 4 == 0:
                agent.update_policy()

        if log:
            agent.log(state, action)

        ep_return += (gamma**t) * reward
        undiscounted_ep_return += reward

        state = next_state
        done = terminated or truncated
        t += 1

    return ep_return, undiscounted_ep_return, t


if __name__ == "__main__":
    ex.run_commandline()
