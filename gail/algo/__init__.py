from .ppo import PPO
from .sac import SAC
from .sac_exp import SACExp, SACInference
from .gail import EssInfoGAIL

ALGOS = {
    'Ess-InfoGAIL': EssInfoGAIL
}
