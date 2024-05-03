from discrete_agent.q_learning import QLearning
from discrete_agent.qtdl import QTDL
from discrete_agent.opt_qtdl import OptQTDL
from discrete_agent.ctdl import CTDL
from discrete_agent.opt_ctdl import OptCTDL


DISCRETE_AGENTS = {
    "q-learning": QLearning,
    "qtdl": QTDL,
    "opt-qtdl": OptQTDL,
    "ctdl": CTDL,
    "opt-ctdl": OptCTDL,
}