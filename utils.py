import gymnasium as gym
from typing import Tuple
import numpy as np
from agent.agent import Agent


def run_episode(env: gym.Env, agent: Agent, gamma: float, train=True, log_arg=None) -> Tuple[np.ndarray, np.ndarray, int]:
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

    done = np.array([False] * env.num_envs)
    while not np.all(done):
        action = agent.act(state, train)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if train:
            for i in range(env.num_envs):
                if not done[i]:
                    agent.replay_buffer.push(state[i], action[i], reward[i], terminated[i])

            if agent.replay_buffer.is_ready():
                agent.update_policy()

        if log_arg is not None:
            agent.log(log_arg)

        ep_return += (gamma**t) * reward
        undiscounted_ep_return += reward
        t += int(np.logical_not(done).sum())

        state = next_state
        done = done | terminated | truncated

    return ep_return, undiscounted_ep_return, t
