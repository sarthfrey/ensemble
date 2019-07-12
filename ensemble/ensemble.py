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

  def __init__(self, name, children=[], weights=None):
    self.name = name
    Ensemble._raise_if_invalid_ensemble_name(name)
    Ensemble._raise_if_invalid_weights(weights, children)
    Ensemble.ensembles[self.name] = self
    self._add_models(children, weights)
    self.weights = self._get_weights()
    self.children = self._get_children()

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)

  def __repr__(self):
    m = ''.join(f'    \'{k}\': {pprint.pformat(v)},\n' for k, v in self.generate_children())
    if self.get_weights() is None:
      w = None
    else:
      w = '[\n' + ''.join(f'    \'{weight}\',\n' for weight in self.get_weights()) + '  ]'
    return (
      'Ensemble(\n'
      f'  name=\'{self.name}\',\n'
      '  children={\n'
      f'{m}'
      '  },\n'
      f'  weights={w},\n'
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
    if 'child' not in kwargs:
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

  def _get_children(self):
    return {
      name: func for name, func in Ensemble.model_functions.items() if self.name in Ensemble.ensemble_groups[name]
    }

  def _add_models(self, model_functions, weights):
    for i, model_function in enumerate(model_functions):
      weight = None if weights is None else weights[i]
      Ensemble.add_model(model_function, self.name, weight)

  def get_weights(self):
    return self.weights

  def set_weights(self, weights):
    self.weights = weights

  def get_children(self):
    return self.children

  def call(self, *args, **kwargs):
    Ensemble._raise_if_invalid_ensemble_kwargs(kwargs)
    child_model_name = kwargs.get('child')
    kwargs.pop('child', None)
    Ensemble._raise_if_model_not_found(child_model_name)
    self._raise_if_model_not_in_ensemble(child_model_name)
    child_model_function = self.children[child_model_name]
    return child_model_function(*args, **kwargs)

  def generate_children(self):
    for model_name, model_function in self.children.items():
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
    return agg(self.get_all_call_return_values(**kwargs))

  def mean(self, **kwargs):
    return self.aggregate(np.mean, **kwargs)

  def sum(self, **kwargs):
    return self.aggregate(np.sum, **kwargs)

  def weighted_mean(self, **kwargs):
    agg = partial(np.average, weights=self._get_weights())
    return self.aggregate(agg, **kwargs)

  def weighted_sum(self, **kwargs):
    agg = lambda values: np.dot(values, self._get_weights())
    return self.aggregate(agg, **kwargs)

  def vote(self, **kwargs):
    return self.aggregate(np.bincount, **kwargs).argmax()
