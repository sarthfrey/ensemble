import pprint
import json
import inspect
import numpy as np

from collections import defaultdict
from functools import partial


class Ensemble(object):
  model_functions = {}
  ensemble_groups = defaultdict(set)
  arg_names = defaultdict(set)
  weights = defaultdict(dict)

  def __init__(self, name, model_fns=[], weights=None):
    self.name = name
    if not name or not isinstance(name, str):
      raise ValueError('Ensemble name must be a non-empty string')
    if weights and len(weights) != len(model_fns):
      raise ValueError('Number of weights must be equal to number of model functions if weights are specified')
    for i, model_function in enumerate(model_fns):
      self.model_functions[model_function.__name__] = model_function
      self.ensemble_groups[model_function.__name__] |= set([self.name])
      self.arg_names[model_function.__name__] |= set(inspect.getfullargspec(model_function)[0])
      if weights:
        self.weights[self.name][model_function.__name__] = weights[i]

  def __call__(self, *args, **kwargs):
    if 'model' not in kwargs:
      return ValueError('Ensemble object must be called with `model` argument')
    model_name = kwargs.get('model')
    self._raise_if_model_not_found(model_name)
    self._raise_if_model_not_in_ensemble(model_name)
    model_function = self.model_functions[model_name]
    kwargs.pop('model', None)
    return model_function(*args, **kwargs)

  def __repr__(self):
    m = ''.join(f'    \'{k}\': {pprint.pformat(v)}\n' for k, v in self.generate_model_functions())
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

  def _raise_if_model_not_found(self, model_name):
    if model_name not in self.model_functions or model_name not in self.ensemble_groups:
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
    return list(self.weights[self.name].values()) if self.weights else None

  def generate_model_functions(self):
    for model_name, model_function in self.model_functions.items():
      if self.name in self.ensemble_groups[model_name]:
        yield model_name, model_function

  def generate_all_calls(self, **kwargs):
    for model_name, model_function in self.generate_model_functions():
      filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.arg_names[model_name]}
      yield model_name, model_function(**filtered_kwargs)

  def generate_all_call_return_values(self, **kwargs):
    return (return_value for _, return_value in self.generate_all_calls(**kwargs))

  def all(self, **kwargs):
    return {k: v for k, v in self.generate_all_calls(**kwargs)}

  def aggregate(self, agg, **kwargs):
    return agg(self.generate_all_call_return_values(**kwargs))

  def apply(self, app, **kwargs):
    return app(list(self.generate_all_call_return_values(**kwargs)))

  def mean(self, **kwargs):
    return self.apply(np.mean, **kwargs)

  def sum(self, **kwargs):
    return self.apply(np.sum, **kwargs)

  def weighted_mean(self, **kwargs):
    app = partial(np.average, weights=self._get_weights())
    return self.apply(app, **kwargs)

  def weighted_sum(self, **kwargs):
    return np.dot(list(self.generate_all_call_return_values(**kwargs)), self._get_weights())

