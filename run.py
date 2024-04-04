from sacred import Experiment
import gymnasium as gym
from time import sleep
from agent import AGENTS

ex = Experiment()

@ex.config
def config():
    env_name = "CliffWalking-v0"
    agent_id = "random"
    agent_config = {}
    num_trajectories = 1000

@ex.automain
def main(env_name, agent_id, agent_config, num_trajectories):
    env = gym.make(env_name, render_mode="human")

    agent = AGENTS[agent_id](env.observation_space, env.action_space)
    agent.setup_agent(agent_config)

    for _ in range(num_trajectories):
        state, info = env.reset()
        done = False
        while not done:
            action = agent.act(state, train=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.replay_buffer.put((state, action, reward, next_state))

            state = next_state
            done = terminated or truncated

    env.close()
