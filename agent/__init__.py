from agent.agent import RandomAgent
from agent.dqn import AtariDQNAgent
from agent.iqn import AtariIQNAgent
from agent.tabular_q_learning import TabularQLearningAgent

AGENTS = {
    "random": RandomAgent,
    "atari-dqn": AtariDQNAgent,
    "atari-iqn": AtariIQNAgent,
    "tabular-q-learning": TabularQLearningAgent,
}
