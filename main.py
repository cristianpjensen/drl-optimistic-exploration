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
    env_config = {}

    # By using 4 environments, it is the same as updating every 4th timestep, as in
    # (https://www.nature.com/articles/nature14236). Make sure if you use more environments, you
    # update the batch size accordingly (e.g. if num_envs=8, batch_size=64).
    num_envs = 4
    batch_size = 32
    agent_id = "atari-dqn"
    min_buffer_size = 100
    max_buffer_size = 1_000_000
    train_steps = 50_000_000
    test_episodes = 5

    train_on_current_transition = False

    # Every 10 training episodes, run a test episode
    train_test_interval = 10

    # Default: 0.99 (https://www.nature.com/articles/nature14236).
    gamma = 0.99


@ex.main
def main(
    env_name: str,
    env_config: dict,
    num_envs: int,
    batch_size: int,
    agent_id: str,
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

    timesteps_trained = 0
    episodes_trained = 0

    # TODO: Make more efficient by not waiting for all environments to be done, but rather continue.

    with tqdm(total=train_steps, desc="Training") as pbar:
        while timesteps_trained < train_steps:
            discounted_return, undiscounted_return, timesteps = run_episode(envs, agent, gamma, train=True, log_arg=run)
            timesteps_trained += timesteps

            if episodes_trained % train_test_interval == 0:
                run["train/test_discounted_return"].append(step=timesteps_trained, value=discounted_return.mean())
                run["train/test_undiscounted_return"].append(step=timesteps_trained, value=undiscounted_return.mean())

            run["train/discounted_return"].append(step=timesteps_trained, value=discounted_return.mean())
            run["train/undiscounted_return"].append(step=timesteps_trained, value=undiscounted_return.mean())

            pbar.update(timesteps)
            episodes_trained += 1

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

    for ep in tqdm(range(test_episodes), desc="Testing"):
        discounted_return, undiscounted_return, _ = run_episode(envs, agent, gamma, train=False)
        for i in range(envs.num_envs):
            run["test/discounted_return"].append(discounted_return[i])
            run["test/undiscounted_return"].append(undiscounted_return[i])

        if ep == 0:
            run["test/video"].upload(f"videos/test-episode-{ep}.mp4", wait=True)

    envs.close()

    saved = agent.save("weights")
    if saved:
        shutil.make_archive("weights", "zip", "weights")
        run["weights"].upload("weights.zip", wait=True)
        shutil.rmtree("weights")
        os.remove("weights.zip")

    # Cleanup
    shutil.rmtree("videos")


def run_episode(
    env: gym.Env,
    agent: Agent,
    gamma: float,
    train=True,
    train_on_current_transition=False,
    log_arg=None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run an episode in the environment with the given agent.

    Args:
        env: Environment.
        agent: Agent.
        gamma: Discount factor.
        train: If enabled, the transitions are added to the replay buffer, and
            the agent's policy is updated each iteration.
        train_on_current_transition: If enabled, the agent is trained on the
            current transition only, and not from the replay buffer.
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
    timestep = 0
    actions = 0

    done = np.array([False] * env.num_envs)
    while not np.all(done):
        action = agent.act(state, train)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if train:
            for i in range(env.num_envs):
                if not done[i]:
                    agent.replay_buffer.push(state[i], action[i], reward[i], terminated[i])

            if train_on_current_transition:
                state_ = torch.tensor(np.array(state))
                action_ = torch.tensor(np.array(action))
                reward_ = torch.tensor(np.array(reward))
                next_state_ = torch.tensor(np.array(next_state))
                terminated_ = torch.tensor(np.array(terminated))
                agent.train(state_, action_, reward_, next_state_, terminated_)

            elif agent.replay_buffer.is_ready():
                agent.update_policy()

        if log_arg is not None:
            agent.log(log_arg)

        ep_return += (gamma**timestep) * reward
        undiscounted_ep_return += reward
        timestep += 1
        actions += int(np.logical_not(done).sum())

        state = next_state
        done = done | terminated | truncated

    return ep_return, undiscounted_ep_return, actions


if __name__ == "__main__":
    ex.run_commandline()
