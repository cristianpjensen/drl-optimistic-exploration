from agent.agent import RandomAgent
from agent.dqn import AtariDQNAgent
from agent.tabular_q_learning import TabularQLearningAgent

AGENTS = {
    "random": RandomAgent,
    "atari-dqn": AtariDQNAgent,
    "tabular-q-learning": TabularQLearningAgent,
}
