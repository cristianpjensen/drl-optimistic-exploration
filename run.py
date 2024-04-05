import os
import shutil
import torch
from dotenv import load_dotenv
from tqdm import tqdm
import gymnasium as gym
from sacred import Experiment
import neptune
from neptune.integrations.sacred import NeptuneObserver
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

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
    train_episodes = 50
    test_episodes = 5
    gamma = 0.99
    agent_config = {
        "gamma": gamma,
    }


@ex.main
def main(
    env_name: str,
    agent_id: str,
    agent_config: dict,
    train_episodes: int,
    test_episodes: int,
    gamma: float,
):
    env = gym.make(env_name, render_mode="rgb_array")

    # Preprocessing for Atari games. Make sure to only use the ALE namespace, such that this
    # condition does not fail.
    if env.spec.namespace == "ALE":
        # The ALE environments already have frame skipping
        env = AtariPreprocessing(env, frame_skip=1, screen_size=84)
        env = FrameStack(env, 4)

    # Normalization
    env = NormalizeObservation(env)
    env = NormalizeReward(env)

    agent = AGENTS[agent_id](env.observation_space, env.action_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.setup(agent_config, device)

    for _ in tqdm(range(train_episodes), desc="Training"):
        ep_return = run_episode(env, agent, gamma, train=True, log=False)
        run["train/episode_return"].append(ep_return)

    # Add video recorder wrapper only on the first test episode
    env.reset()
    env = RecordVideo(env, "videos", name_prefix="test", disable_logger=True, episode_trigger=lambda t: t == 0)

    for ep in tqdm(range(test_episodes), desc="Testing"):
        ep_return = run_episode(env, agent, gamma, train=False, log=True)
        run["test/episode_return"].append(ep_return)

        if ep == 0:
            run["test/video"].upload(f"videos/test-episode-{ep}.mp4", wait=True)

    env.close()

    # Remove videos
    shutil.rmtree("videos")


def run_episode(env: gym.Env, agent: Agent, gamma, train=True, log=False):
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
        The (discounted) return of the episode.

    """

    state, _ = env.reset()
    ep_return = 0
    timestep = 0

    done = False
    while not done:
        action = agent.act(state, timestep, train)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if train:
            agent.replay_buffer.push(state, action, reward, next_state)

            if agent.replay_buffer.is_ready():
                agent.update_policy()

        if log:
            agent.log(state, action)

        ep_return += (gamma ** timestep) * reward
        state = next_state
        done = terminated or truncated
        timestep += 1

    return ep_return


if __name__ == "__main__":
    ex.run_commandline()
