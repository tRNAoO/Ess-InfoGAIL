from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import EssInfoGAIL

ALGOS = {
    'Ess-InfoGAIL': EssInfoGAIL
}
