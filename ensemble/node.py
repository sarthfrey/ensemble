import inspect

from abc import ABC, abstractmethod
from collections import defaultdict


class Node(ABC):
  model_functions = {}                # model_name    -> model
  arg_names = defaultdict(set)        # model_name    -> arg_names
  ensemble_groups = defaultdict(set)  # model_name    -> ensemble_names
  ensembles = dict()                  # ensemble_name -> ensemble
  weight_map = defaultdict(dict)      # ensemble_name -> model_name -> weight

  @abstractmethod
  def __init__(self, *args, **kwargs):
    pass

  @abstractmethod
  def __call__(self, *args, **kwargs):
    pass

  @classmethod
  def add_model(cls, ensemble_name, model_function, weight=None):
    print(model_function)
    cls.model_functions[model_function.__name__] = model_function
    cls.ensemble_groups[model_function.__name__] |= set([ensemble_name])
    cls.arg_names[model_function.__name__] |= set(inspect.getfullargspec(model_function)[0])
    if weight is not None:
      cls.weight_map[ensemble_name][model_function.__name__] = weight

  @classmethod
  def add_models(cls, ensemble_name, model_functions, weights):
    for i, model_function in enumerate(model_functions):
      weight = None if weights is None else weights[i]
      cls.add_model(ensemble_name, model_function, weight)

  @classmethod
  def _get_children(cls, ensemble_name):
    return {
      name: func for name, func in cls.model_functions.items() if ensemble_name in cls.ensemble_groups[name]
    }

  @classmethod
  def _get_weights(cls, ensemble_name):
    return list(cls.weight_map[ensemble_name].values()) if cls.weight_map[ensemble_name] else None  
