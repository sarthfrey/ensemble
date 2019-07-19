import pprint
import json
import numpy as np

from functools import partial
from .node import Node
from .graph import Graph
from .model import Model
from .poller import poller


class Ensemble(Node):
  """
  A user created :class:`Ensemble <Ensemble>` object that can
  multiplex children, call all of them, and aggregate results.

  :param name: required `str` to identify the ensemble to its :class:`Graph <Graph>`
  :param children: optional `list` of :class:`Node <Graph>` objects
  :param weights: optional `list` of `float` objects per child
  :param mode: optional `str` that specifies what the ensemble does when it is called
  """
  POLLING_STRAGIES = {
    'structured',
    'flat',
  }
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

  # error helpers

  @classmethod
  def _raise_if_invalid_init(cls, name, children, weights, mode):
    cls._raise_if_invalid_ensemble_name(name)
    cls._raise_if_invalid_weights(weights, children)
    cls._raise_if_invalid_mode(mode)

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
  def _raise_if_node_not_found(cls, node_name):
    if node_name not in Graph.nodes:
      raise ValueError(
        f'There is no node with name `{node_name}` in the graph'
      )

  @staticmethod
  def _raise_if_node_not_in_ensemble(ensemble_name, node_name):
    if ensemble_name not in Graph.ensemble_groups[node_name]:
      raise ValueError(
        f'Node `{node_name}` is not a child of Ensemble `{ensemble_name}`'
      )

  # properties

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

  def set_polling_strategy(self, polling_strategy):
    if polling_strategy not in self.POLLING_STRAGIES:
      raise f'`{polling_strategy}` is not a valid polling strategy'
    self.polling_strategy = polling_strategy

  def get_polling_strategy(self):
    return self.polling_strategy

  # child polling helpers

  def generate_children(self):
    for name, node in self.children.items():
      yield name, node

  def generate_all_calls(self, arg_dict, **kwargs):
    for name, node in self.generate_children():
      if arg_dict:
        yield name, node(**arg_dict[name])
      else:
        if isinstance(node, self.__class__):
          yield name, node(**kwargs)
        else:
          filtered_kwargs = {k: v for k, v in kwargs.items() if k in node.get_arg_names()}
          yield name, node(**filtered_kwargs)

  def generate_all_call_return_values(self, arg_dict, **kwargs):
    return (return_value for _, return_value in self.generate_all_calls(arg_dict, **kwargs))

  def get_all_call_return_values(self, arg_dict, **kwargs):
    return list(self.generate_all_call_return_values(arg_dict, **kwargs))

  # main callers

  def multiplex(self, child, **kwargs):
    Ensemble._raise_if_node_not_found(child)
    Ensemble._raise_if_node_not_in_ensemble(self.name, child)
    child = self.children[child]
    return child(**kwargs)

  @poller
  def all(self, arg_dict=dict(), **kwargs):
    return {k: v for k, v in self.generate_all_calls(arg_dict, **kwargs)}

  @poller
  def aggregate(self, agg, arg_dict=dict(), **kwargs):
    return agg(self.get_all_call_return_values(arg_dict, **kwargs))

  @poller
  def mean(self, arg_dict=dict(), **kwargs):
    return self.aggregate(np.mean, arg_dict, **kwargs)

  @poller
  def sum(self, arg_dict=dict(), **kwargs):
    return self.aggregate(np.sum, arg_dict, **kwargs)

  # other callers

  @poller
  def weighted_mean(self, arg_dict=dict(), **kwargs):
    agg = partial(np.average, weights=self.weights)
    return self.aggregate(agg, arg_dict, **kwargs)

  @poller
  def weighted_sum(self, arg_dict=dict(), **kwargs):
    agg = lambda values: np.dot(values, self.weights)
    return self.aggregate(agg, arg_dict, **kwargs)

  @poller
  def vote(self, arg_dict=dict(), **kwargs):
    return self.aggregate(np.bincount, arg_dict, **kwargs).argmax()
