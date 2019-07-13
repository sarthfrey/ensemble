import pprint
import json
import numpy as np

from functools import partial
from .node import Node
from .graph import Graph
from .model import Model


class Ensemble(Node):
  MODES = {
    'multiplex',
    'all',
    'aggregate',
    'sum',
    'mean',
  }
  DEFAULT_MODE = 'all'

  def __init__(self, name, children=[], weights=None, mode=DEFAULT_MODE):
    Ensemble._raise_if_invalid_init(name, children, weights, mode)
    self.name = name
    self.mode = mode
    self._init_to_graph(children, weights)
    self.children = Graph._get_children(self.name)
    self.weights = Graph._get_weights(self.name)

  def __call__(self, *args, **kwargs):
    return getattr(self, self.get_mode())(*args, **kwargs)

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
    # Graph.ensembles[self.name] = self
    for i, child in enumerate(children):
      weight = None if weights is None else weights[i]
      if callable(child) and not isinstance(child, Ensemble):
        child = Model(child, self.name)
      Graph.add_node(self.name, child, weight)

  @classmethod
  def _raise_if_invalid_init(cls, name, children, weights, mode):
    cls._raise_if_invalid_ensemble_name(name)
    cls._raise_if_invalid_weights(weights, children)
    cls._raise_if_invalid_mode(mode)

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
  def _raise_if_invalid_mode(cls, mode):
    if mode not in cls.MODES:
      raise ValueError('Number of weights must be equal to number of child models if weights are specified')

  @classmethod
  def _raise_if_model_not_found(cls, model_name):
    if model_name not in Graph.nodes or model_name not in Graph.nodes:
      raise ValueError(
        f'Either there is no decorated model function `{model_name}` or it was not added to the Ensemble'
      )

  @staticmethod
  def _raise_if_model_not_in_ensemble(ensemble_name, model_name):
    ensemble_group = Graph.ensemble_groups[model_name]
    if ensemble_name not in ensemble_group:
      raise ValueError(
        f'Model function `{model_name}` is not attached to ensemble `{ensemble_name}`'
      )

  def get_name(self):
    return self.name

  def get_mode(self):
    return self.mode

  def set_mode(self, mode):
    self.mode = mode
    return self

  def get_weights(self):
    return self.weights

  def set_weights(self, weights):
    self.weights = weights

  def get_children(self):
    return self.children

  def multiplex(self, *args, **kwargs):
    Ensemble._raise_if_invalid_call_kwargs(kwargs)
    child_name = kwargs.get('child')
    kwargs.pop('child', None)
    Ensemble._raise_if_model_not_found(child_name)
    Ensemble._raise_if_model_not_in_ensemble(self.name, child_name)
    child = self.children[child_name]
    return child(*args, **kwargs)

  def generate_children(self):
    for name, node in self.children.items():
      yield name, node

  def generate_all_calls(self, **kwargs):
    for name, node in self.generate_children():
      if isinstance(node, self.__class__):
        yield name, node(**kwargs)
      else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in node.get_arg_names()}
        yield name, node(**filtered_kwargs)

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
