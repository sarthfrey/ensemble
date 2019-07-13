import pprint
import json
import inspect
import numpy as np

from functools import partial
from .node import Node


class Ensemble(Node):

  def __init__(self, name, children=[], weights=None):
    self.name = name
    self._raise_if_invalid_init(name, children, weights)
    self._init_to_graph(children, weights)
    self.children = super()._get_children(self.name)
    self.weights = super()._get_weights(self.name)

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

  def _init_to_graph(self, children, weights):
    super().ensembles[self.name] = self
    super().add_models(self.name, children, weights)

  def _raise_if_invalid_init(self, name, children, weights):
    Ensemble._raise_if_invalid_ensemble_name(name)
    Ensemble._raise_if_invalid_weights(weights, children)

  @staticmethod
  def _raise_if_invalid_call_kwargs(kwargs):
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
    if model_name not in super().model_functions or model_name not in super().ensemble_groups:
      raise ValueError(
        f'Either there is no decorated model function `{model_name}` or it was not added to the Ensemble'
      )

  def _raise_if_model_not_in_ensemble(self, model_name):
    ensemble_group = super().ensemble_groups[model_name]
    if self.name not in ensemble_group:
      raise ValueError(
        f'Model function `{model_name}` is not attached to ensemble `{self.name}`'
      )

  def get_weights(self):
    return self.weights

  def set_weights(self, weights):
    self.weights = weights

  def get_children(self):
    return self.children

  def call(self, *args, **kwargs):
    Ensemble._raise_if_invalid_call_kwargs(kwargs)
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
      filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.arg_names[model_name]}
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
    agg = partial(np.average, weights=self.weights)
    return self.aggregate(agg, **kwargs)

  def weighted_sum(self, **kwargs):
    agg = lambda values: np.dot(values, self.weights)
    return self.aggregate(agg, **kwargs)

  def vote(self, **kwargs):
    return self.aggregate(np.bincount, **kwargs).argmax()
