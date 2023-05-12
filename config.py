# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.BATCH_SIZE = 128
# The all important scales for the stuff
_C.TRAIN.CONTEXT_SIZE = 1024
_C.TRAIN.N_LAYERS = 6
_C.TRAIN.D_MODEL = 256
_C.TRAIN.N_EPOCH = 1


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`