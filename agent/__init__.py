from agent.agent import RandomAgent
from agent.dqn import AtariDQNAgent
from agent.tabular_q_learning import TabularQLearningAgent
from agent.qr_dqn import AtariQRAgent

AGENTS = {
    "random": RandomAgent,
    "tabular-q-learning": TabularQLearningAgent,
    "atari-dqn": AtariDQNAgent,
    "atari-qr-dqn": AtariQRAgent,
}
