# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.CLASS_NUMBER = 21
# The all important scales for the stuff
_C.TRAIN.BATCH_SIZE = 3
_C.TRAIN.LR = 0.01
_C.TRAIN.EPOCHS = 120
_C.TRAIN.HIDDEN_LAYER = 256
_C.TRAIN.TRAINABLE = 3
_C.TRAIN.ISTEST = False
_C.TRAIN.LR_STEP = 20
_C.TRAIN.LR_GAMMA = 0.9
_C.TRAIN.ISNORM = True


def get_cfg_defaults():
    """ Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

cfg = _C
