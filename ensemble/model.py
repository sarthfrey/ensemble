import inspect

from .node import Node
from .graph import Graph
from typing import Callable, Set, Optional, List


class Model(Node):
  """
  An object that represents a callable model and that may be attached to
  :class:`Ensemble <Ensemble>` objects, without interfering with the
  underlying model function
  """
  invalid_args_names = [
  ]

  def __init__(
    self,
    name: str,
    call: Callable,
    ensemble_names: List[str] = [],
    is_function: bool = False,
  ):
    self.name = name
    self.model_function = call
    self.ensemble_names = set(ensemble_names)
    self.is_function = is_function
    self.arg_names = set(inspect.getfullargspec(call)[0]) if is_function else None
    if any(not ensemble_name for ensemble_name in self.ensemble_names):
      raise ValueError('Must provide a valid ensemble names')
    for ensemble_name in ensemble_names:
      Graph.add_node(ensemble_name, self)

  def __call__(self, *args, **kwargs):
    return self.model_function(*args, **kwargs)

  def __repr__(self) -> str:
    return (
      f"Model(name='{self.name}', func={self.name}({', '.join(self.arg_names)}))"
    )

  def __str__(self):
    return self._str()

  def _str(self, level: int = 0) -> str:
    return '\t' * level + repr(self) + "\n"

  def get_name(self) -> str:
    return self.name

  def get_arg_names(self) -> Set[str]:
    return self.arg_names

def child(*ensemble_names: str) -> Callable:
  """
  A decorator used to attach a model function to ensembles

  :param ensemble_names: an unpacked list of ensemble names to attach the model to
  :return: a wrapper function that decorates the function as model
  :rtype: :class:`Model <Model>` object
  """
  def wrapper(model_function: Callable) -> Model:
    return Model(
      name=model_function.__name__,
      call=model_function,
      ensemble_names=[*ensemble_names],
      is_function=True
    )
  return wrapper
