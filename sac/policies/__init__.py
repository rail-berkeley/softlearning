try:
    from .nn_policy import NNPolicy
    from .gmm import GMMPolicy
except Exception as e:
    pass
from .real_nvp import RealNVPPolicy
