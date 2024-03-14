from collections import namedtuple, deque
from old.parameters import *
import random


Transition = namedtuple("Transition", ["States", "Actions", "Rewards", "NextStates", "Dones"])


class ReplayMemory:
  """
  Implementation of Agent memory
  """
  def __init__(self, capacity=MEMORY_LEN):
    self.memory = deque(maxlen=capacity)

  def store(self, t):
    self.memory.append(t)

  def sample(self, n):
    a = random.sample(self.memory, n)
    return a

  def __len__(self):
    return len(self.memory)