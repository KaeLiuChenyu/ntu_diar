from enum import Enum, auto


#---------------------------------------------------------#
class Stage(Enum):
#---------------------------------------------------------#
  """
  Simple enum to track stage of experiments.
  """
  TRAIN = auto()
  VALID = auto()
  TEST = auto()