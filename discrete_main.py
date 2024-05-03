import os
import random
import shutil
import string
from glob import glob

import gymnasium as gym
import neptune
from dotenv import load_dotenv
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.time_limit import TimeLimit
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment
from tqdm import tqdm

from discrete_agent import DISCRETE_AGENTS
from discrete_agent.discrete_agent import DiscreteAgent

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
    env_name = "CliffWalking-v0"
    env_config = {}

    agent_id = "ctdl"
    train_steps = 100_000
    gamma = 0.85

    record_video_every = 100


@ex.main
def main(
    env_name: str,
    env_config: dict,
    agent_id: str,
    train_steps: int,
    gamma: float,
    record_video_every: int,
):
    NAME = get_random_id()
    os.mkdir(NAME)

    print("Run ID:", NAME)
    print("DO NOT DELETE THIS DIRECTORY!")

    env = gym.make(env_name, render_mode="rgb_array", **env_config)
    env = TimeLimit(env, max_episode_steps=500)
    env = RecordVideo(env, f"{NAME}/videos", disable_logger=True, episode_trigger=lambda t: t % record_video_every == 0)
    agent = DISCRETE_AGENTS[agent_id](env.observation_space, env.action_space)

    config = { "gamma": gamma }
    if hasattr(env.observation_space, "n"):
        config["n_states"] = env.observation_space.n

    if hasattr(env.action_space, "n"):
        config["n_actions"] = env.action_space.n

    agent.setup(config)
    env.metadata["render_fps"] = 30

    train_discrete_agent(env, agent, gamma, train_steps, run)

    env.close()

    for i, video in enumerate(sorted(glob(f"{NAME}/videos/*.mp4"))):
        run[f"video/episode-{i*record_video_every}.mp4"].upload(video, wait=True)

    # Save weights
    os.mkdir(f"{NAME}/weights")
    saved = agent.save(f"{NAME}/weights")
    if saved:
        shutil.make_archive(f"{NAME}/weights", "zip", f"{NAME}/weights")
        run["weights"].upload(f"{NAME}/weights.zip", wait=True)

    # Clean up
    shutil.rmtree(NAME)


def train_discrete_agent(
    env: gym.Env,
    agent: DiscreteAgent,
    gamma: float,
    total_steps: int,
    run: neptune.Run,
):
    state, _ = env.reset()

    ep_timesteps = 0
    ep_return = 0
    ep_discounted_return = 0

    for step in tqdm(range(total_steps)):
        action = agent.act(state, train=True)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.update_policy(state, action, reward, next_state, terminated)

        state = next_state

        # Update statistics
        ep_return += reward
        ep_discounted_return += (gamma ** ep_timesteps) * reward
        ep_timesteps += 1

        # Reset environment
        if terminated or truncated:
            run["train/discounted_return"].append(step=step, value=ep_discounted_return)
            run["train/undiscounted_return"].append(step=step, value=ep_return)
            run["train/episode_timesteps"].append(step=step, value=ep_timesteps)

            ep_return = 0
            ep_discounted_return = 0
            ep_timesteps = 0

            state, _ = env.reset()

        agent.log(run)


def get_random_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


if __name__ == "__main__":
    ex.run_commandline()
