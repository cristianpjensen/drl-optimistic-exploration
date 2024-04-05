from typing import Dict
from agent.agent import RandomAgent
from agent.dqn import AtariDQNAgent


AGENTS = {
    "random": RandomAgent,
    "atari-dqn": AtariDQNAgent,
}
