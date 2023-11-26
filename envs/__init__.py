from .traj import Traj
from .reacher import MultimodalReacher
from .pusher import MultimodalPusher
from .walker import MultimodalWalker
from .humanoid import MultimodalHumanoid


MultimodalEnvs = {
    'Reacher-v4': MultimodalReacher,
    '2D-Trajectory': Traj,
    'Pusher-v4': MultimodalPusher,
    'Walker2d-v4': MultimodalWalker,
    'Humanoid-v4': MultimodalHumanoid
}
