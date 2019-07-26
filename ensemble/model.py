import inspect

from .node import Node
from .graph import Graph
from typing import Callable, Set


class Model(Node):
  """
  An object that represents a callable model and that may be attached to
  :class:`Ensemble <Ensemble>` objects, without interfering with the
  underlying model function
  """
  invalid_args_names = [
    'model',
  ]

  def __init__(self, model_function: Callable, *ensemble_names: str):
    self.name = model_function.__name__
    self.model_function = model_function
    self.ensemble_names = set(ensemble_names)
    self.arg_names = set(inspect.getfullargspec(model_function)[0])
    if any(not ensemble_name for ensemble_name in self.ensemble_names):
      raise ValueError('Must provide a valid ensemble names')
    for ensemble_name in ensemble_names:
      Graph.add_node(ensemble_name, self)
    Model._validate_model_function(model_function)

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

  @staticmethod
  def _validate_model_function(model_function):
    arg_names = inspect.getfullargspec(model_function)[0]
    for invalid_arg_name in Model.invalid_args_names:
      if invalid_arg_name in arg_names:
        raise ValueError(
          f'Function `{model_function.__name__}` is decorated with @model '
          f'and so it may not have `{invalid_arg_name}` as an argument'
        )

def child(*ensemble_names: str) -> Callable:
  """
  A decorator used to attach a model function to ensembles

  :param ensemble_names: an unpacked list of ensemble names to attach the model to
  :return: a wrapper function that decorates the function as model
  :rtype: :class:`Model <Model>` object
  """
  def wrapper(model_function: Callable) -> Model:
    return Model(model_function, *ensemble_names)
  return wrapper
