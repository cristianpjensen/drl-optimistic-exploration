from agent.agent import RandomAgent
from agent.dqn import AtariDQNAgent
from agent.c51 import AtariC51Agent
from agent.opt_c51 import AtariOptC51Agent
from agent.bayesian_opt_c51 import AtariBayesianOptC51Agent
from agent.iqn import AtariIQNAgent
from agent.opt_iqn import AtariOptIQNAgent
from agent.bayesian_opt_iqn import AtariBayesianOptIQNAgent
from agent.qr_dqn import AtariQRAgent
from agent.opt_qr_dqn import AtariOptQRAgent
from agent.bayesian_opt_qr_dqn import AtariBayesianOptQRAgent

AGENTS = {
    "random": RandomAgent,
    "atari-dqn": AtariDQNAgent,
    "atari-iqn": AtariIQNAgent,
    "atari-opt-iqn": AtariOptIQNAgent,
    "atari-bayesian-opt-iqn": AtariBayesianOptIQNAgent,
    "atari-qr-dqn": AtariQRAgent,
    "atari-opt-qr-dqn": AtariOptQRAgent,
    "atari-bayesian-opt-qr-dqn": AtariBayesianOptQRAgent,
    "atari-c51": AtariC51Agent,
    "atari-opt-c51": AtariOptC51Agent,
    "atari-bayesian-opt-c51": AtariBayesianOptC51Agent,
}
