import inspect

from abc import ABC, abstractmethod
from collections import defaultdict


class Node(ABC):
  @abstractmethod
  def __init__(self, *args, **kwargs):
    pass

  @abstractmethod
  def __call__(self, *args, **kwargs):
    pass

  @abstractmethod
  def __repr__(self, *args, **kwargs):
    pass

  @abstractmethod
  def get_name(self):
    pass

"""
  @abstractmethod
  def set_name(self, name):
    pass
"""
