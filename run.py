import os
import shutil

import gymnasium as gym
import neptune
import torch
from dotenv import load_dotenv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.normalize import NormalizeReward
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.frame_stack import FrameStack
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment
from tqdm import tqdm
from utils import run_episode

from agent import AGENTS

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
    train_steps = 50_000_000
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
                lambda env: FrameStack(env, 4),
            ],
        )
        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            batch_size=batch_size,
            device=device,
        )
    else:
        envs = gym.vector.make(
            env_name,
            render_mode="rgb_array",
            num_envs=num_envs,
            wrappers=[lambda env: FrameStack(env, 4)]
        )
        agent = AGENTS[agent_id](
            envs.single_observation_space,
            envs.single_action_space,
            batch_size=batch_size,
            device=device,
        )

    envs.metadata["render_fps"] = 30
    agent.setup(agent_config)

    timesteps_trained = 0
    episodes_trained = 0

    with tqdm(total=train_steps, desc="Training") as pbar:
        while timesteps_trained < train_steps:
            discounted_return, undiscounted_return, timesteps = run_episode(envs, agent, gamma, train=True, log_arg=run)
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

    if "ALE" in env_name:
        envs = gym.make_vec(
            env_name,
            render_mode="rgb_array", 
            num_envs=num_envs, 
            wrappers=[
                # The ALE environments already have frame skipping
                lambda env: AtariPreprocessing(env, frame_skip=1, screen_size=84),
                lambda env: RecordVideo(env, "videos", name_prefix="test", disable_logger=True, episode_trigger=lambda t: t == 0),
                NormalizeReward,
            ],
        )
    else:
        envs = gym.vector.make(
            env_name,
            render_mode="rgb_array",
            num_envs=num_envs,
            wrappers=[
                lambda env: RecordVideo(env, "videos", name_prefix="test", disable_logger=True, episode_trigger=lambda t: t == 0),
                NormalizeReward,
            ],
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
        run["weights"].upload("weights.zip", wait=True)
        shutil.rmtree("weights")
        os.remove("weights.zip")

    # Cleanup
    shutil.rmtree("videos")

if __name__ == "__main__":
    ex.run_commandline()
