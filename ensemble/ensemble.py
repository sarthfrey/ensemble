import pprint
import json
import inspect
import numpy as np

from collections import defaultdict
from functools import partial


class Ensemble(object):
  model_functions = {}                # model_name    -> model
  arg_names = defaultdict(set)        # model_name    -> arg_names
  ensemble_groups = defaultdict(set)  # model_name    -> ensemble_names
  ensembles = dict()                  # ensemble_name -> ensemble
  weights = defaultdict(dict)         # ensemble_name -> model_name -> weight

  def __init__(self, name, model_fns=[], weights=None):
    self.name = name
    self.children = dict()
    Ensemble._raise_if_invalid_ensemble_name(name)
    Ensemble._raise_if_invalid_weights(weights, model_fns)
    Ensemble.ensembles[self.name] = self
    for i, model_function in enumerate(model_fns):
      weight = None if weights is None else weights[i]
      Ensemble.add_model(model_function, self.name, weight)
    for model_name, model_function in self.generate_children():
      self.children[model_name] = model_function

  def __call__(self, *args, **kwargs):
    self.call(*args, **kwargs)

  def __repr__(self):
    m = ''.join(f'    \'{k}\': {pprint.pformat(v)}\n' for k, v in self.generate_children())
    return (
      'Ensemble(\n'
      f'  name=\'{self.name}\',\n'
      '  model_functions={\n'
      f'{m}'
      '  }\n'
      ')'
    )

  def __str__(self):
    return self.__repr__()

  @classmethod
  def get_by_name(cls, name):
    return cls.ensembles[name]

  @classmethod
  def add_model(cls, model_function, ensemble_name, weight=None):
    cls.model_functions[model_function.__name__] = model_function
    cls.ensemble_groups[model_function.__name__] |= set([ensemble_name])
    cls.arg_names[model_function.__name__] |= set(inspect.getfullargspec(model_function)[0])
    if weight is not None:
      cls.weights[ensemble_name][model_function.__name__] = weight

  @staticmethod
  def _raise_if_invalid_ensemble_kwargs(kwargs):
    if 'model' not in kwargs:
      raise ValueError('Ensemble object must be called with `model` argument')

  @staticmethod
  def _raise_if_invalid_ensemble_name(ensemble_name):
    if not ensemble_name or not isinstance(ensemble_name, str):
      raise ValueError('Ensemble name must be a non-empty string')

  @staticmethod
  def _raise_if_invalid_weights(weights, children):
    if weights is not None and len(weights) != len(children):
      raise ValueError('Number of weights must be equal to number of child models if weights are specified')

  @classmethod
  def _raise_if_model_not_found(cls, model_name):
    if model_name not in cls.model_functions or model_name not in cls.ensemble_groups:
      raise ValueError(
        f'Either there is no decorated model function `{model_name}` or it was not added to the Ensemble'
      )

  def _raise_if_model_not_in_ensemble(self, model_name):
    ensemble_group = self.ensemble_groups[model_name]
    if self.name not in ensemble_group:
      raise ValueError(
        f'Model function `{model_name}` is not attached to ensemble `{self.name}`'
      )

  def _get_weights(self):
    return list(Ensemble.weights[self.name].values()) if Ensemble.weights[self.name] else None

  def call(self, *args, **kwargs):
    Ensemble._raise_if_invalid_ensemble_kwargs(kwargs)
    child_model_name = kwargs.get('model')
    Ensemble._raise_if_model_not_found(child_model_name)
    self._raise_if_model_not_in_ensemble(child_model_name)
    child_model_function = self.children[child_model_name]
    kwargs.pop('model', None)
    return child_model_function(*args, **kwargs)

  def generate_children(self):
    for model_name, model_function in Ensemble.model_functions.items():
      if self.name in Ensemble.ensemble_groups[model_name]:
        yield model_name, model_function

  def generate_all_calls(self, **kwargs):
    for model_name, model_function in self.generate_children():
      filtered_kwargs = {k: v for k, v in kwargs.items() if k in Ensemble.arg_names[model_name]}
      yield model_name, model_function(**filtered_kwargs)

  def generate_all_call_return_values(self, **kwargs):
    return (return_value for _, return_value in self.generate_all_calls(**kwargs))

  def get_all_call_return_values(self, **kwargs):
    return list(self.generate_all_call_return_values(**kwargs))

  def all(self, **kwargs):
    return {k: v for k, v in self.generate_all_calls(**kwargs)}

  def aggregate(self, agg, **kwargs):
    return agg(self.generate_all_call_return_values(**kwargs))

  def apply(self, app, **kwargs):
    return app(self.get_all_call_return_values(**kwargs))

  def mean(self, **kwargs):
    return self.apply(np.mean, **kwargs)

  def sum(self, **kwargs):
    return self.apply(np.sum, **kwargs)

  def weighted_mean(self, **kwargs):
    app = partial(np.average, weights=self._get_weights())
    return self.apply(app, **kwargs)

  def weighted_sum(self, **kwargs):
    app = lambda values: np.dot(values, self._get_weights())
    return self.apply(app, **kwargs)

