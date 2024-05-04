from agent.agent import RandomAgent
from agent.c51 import AtariC51Agent
from agent.dqn import AtariDQNAgent
from agent.iqn import AtariIQNAgent
from agent.opt_iqn import AtariOptIQNAgent
from agent.opt_qr_dqn import AtariOptQRAgent
from agent.qr_dqn import AtariQRAgent
from agent.tabular_q_learning import TabularQLearningAgent

AGENTS = {
    "random": RandomAgent,
    "tabular-q-learning": TabularQLearningAgent,
    "atari-dqn": AtariDQNAgent,
    "atari-iqn": AtariIQNAgent,
    "atari-opt-iqn": AtariOptIQNAgent,
    "atari-qr-dqn": AtariQRAgent,
    "atari-opt-qr-dqn": AtariOptQRAgent,
    "atari-c51": AtariC51Agent,
}
