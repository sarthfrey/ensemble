import inspect

from abc import ABC, abstractmethod
from collections import defaultdict


class Node(ABC):
  """
  Composite abstract base class for the :class:`Ensemble <Ensemble>` and
  :class:`Model <Model>` classes, owned by sessioned :class:`Graph <Graph>` object
  """
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
  def __str__(self, *args, **kwargs):
    pass

  @abstractmethod
  def get_name(self):
    pass

  def get_arg_names(self):
    return set(inspect.getfullargspec(self)[0])

  @abstractmethod
  def _str(self):
    pass

"""
  @abstractmethod
  def set_name(self, name):
    pass
"""
