import pprint
import json
import numpy as np

from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from .node import Node
from .graph import Graph
from .model import Model
from .poller import poller
from .types import PollingStrategy, CallMode


class Ensemble(Node):
  """
  A user created :class:`Ensemble <Ensemble>` object that can
  multiplex children, call all of them, and aggregate results.

  :param name: required `str` to identify the ensemble to its :class:`Graph <Graph>`
  :param children: optional `list` of :class:`Node <Graph>` objects
  :param weights: optional `list` of `float` objects per child
  :param mode: optional `str` that specifies what the ensemble does when it is called
  """

  def __init__(
    self,
    name: str,
    children: List[Callable] = [],
    weights: Optional[List[np.ndarray]] = None,
    mode: str = CallMode.DEFAULT_MODE.value,
  ):
    Ensemble._raise_if_invalid_init(name, children, weights)
    self.name = name
    self.mode = CallMode(mode)
    self._init_to_graph(children, weights)
    self.children = Graph._get_children(self.name)
    self.weights = Graph._get_weights(self.name)
    self.wrapper = None
    self.child_wrapper = None
    self.child_decorator = None

  def __call__(self, *args, **kwargs):
    ret = getattr(self, self.get_mode())(*args, **kwargs)
    return ret if self.wrapper is None else self.wrapper(ret)

  def __repr__(self) -> str:
    return (
      f"Ensemble(name='{self.name}', children={str(list(self.children.keys()))}, "
      f"weights={str(self.weights)}, mode='{self.get_mode()}')"
    )

  def __str__(self, level: int = 0) -> str:
    ret = '\t' * level + repr(self) + "\n"
    for child in self.children.values():
      ret += child.__str__(level+1)
    return ret

  def _init_to_graph(self, children: List[Callable], weights: Optional[List[np.ndarray]]):
    for i, child in enumerate(children):
      weight = None if weights is None else weights[i]
      if not isinstance(child, Ensemble):
        child = Model(child, self.name)
      Graph.add_node(self.name, child, weight)

  def call_child(self, child_name, **kwargs):
    child = self.children[child_name]
    child = child if self.child_decorator is None else self.child_decorator(child)
    ret = child(**kwargs)
    return ret if self.child_wrapper is None else self.child_wrapper(ret)

  # error helpers

  @classmethod
  def _raise_if_invalid_init(cls, name: str, children: List[Callable], weights: Optional[List[np.ndarray]]):
    cls._raise_if_invalid_ensemble_name(name)
    cls._raise_if_invalid_weights(children, weights)

  @staticmethod
  def _raise_if_invalid_ensemble_name(ensemble_name: str):
    if not ensemble_name:
      raise ValueError('Ensemble name must be a non-empty string')

  @staticmethod
  def _raise_if_invalid_weights(children: List[Callable], weights: Optional[List[np.ndarray]]):
    if weights is not None and len(weights) != len(children):
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

  def get_name(self) -> str:
    return self.name

  def get_wrapper(self) -> Callable:
    return self.wrapper

  def wrap(self, wrapper: Callable):
    self.wrapper = wrapper

  def wrap_children(self, wrapper: Callable):
    self.child_wrapper = wrapper

  def decorate_children(self, decorator: Callable):
    self.child_decorator = decorator

  def get_mode(self) -> str:
    return self.mode.value

  def set_mode(self, mode: str):
    self.mode = CallMode(mode)
    return self

  def get_weights(self) -> Optional[List[np.ndarray]]:
    return self.weights

  def set_weights(self, weights: Optional[List[np.ndarray]]):
    self.weights = weights

  def get_children(self) -> List[Node]:
    return self.children

  def set_polling_strategy(self, polling_strategy: str):
    self.polling_strategy = PollingStrategy(polling_strategy)

  def get_polling_strategy(self) -> str:
    return self.polling_strategy.value

  # child polling helpers

  def generate_children(self) -> Iterator[Tuple[str, Node]]:
    for name, node in self.children.items():
      yield name, node

  def generate_all_calls(self, arg_dict: Dict, **kwargs) -> Iterator[Tuple[str, None]]:
    for name, node in self.generate_children():
      if self.get_polling_strategy() == 'structured':
        yield name, self.call_child(name, **arg_dict[name])
      else:
        if isinstance(node, self.__class__):
          yield name, self.call_child(name, **kwargs)
        else:
          filtered_kwargs = {k: v for k, v in kwargs.items() if k in node.get_arg_names()}
          yield name, self.call_child(name, **filtered_kwargs)

  def generate_all_call_return_values(self, arg_dict: Dict, **kwargs):
    return (return_value for _, return_value in self.generate_all_calls(arg_dict, **kwargs))

  def get_all_call_return_values(self, arg_dict: Dict, **kwargs):
    return list(self.generate_all_call_return_values(arg_dict, **kwargs))

  # main callers

  def multiplex(self, child: str, **kwargs):
    Ensemble._raise_if_node_not_found(child)
    Ensemble._raise_if_node_not_in_ensemble(self.name, child)
    return self.call_child(child, **kwargs)

  @poller
  def call_children(self, arg_dict: Dict = dict(), **kwargs):
    return {k: v for k, v in self.generate_all_calls(arg_dict, **kwargs)}

  @poller
  def aggregate(self, agg: Callable, arg_dict=dict(), **kwargs):
    return agg(self.get_all_call_return_values(arg_dict, **kwargs))

  @poller
  def mean(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    return self.aggregate(np.mean, arg_dict, **kwargs)

  @poller
  def sum(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    return self.aggregate(np.sum, arg_dict, **kwargs)

  @poller
  def max(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    return self.aggregate(max, arg_dict, **kwargs)

  @poller
  def any(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    return self.aggregate(any, arg_dict, **kwargs)

  @poller
  def all(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    return self.aggregate(all, arg_dict, **kwargs)

  # other callers

  @poller
  def weighted_mean(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    agg = partial(np.average, weights=self.weights)
    return self.aggregate(agg, arg_dict, **kwargs)

  @poller
  def weighted_sum(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    agg = lambda values: np.dot(values, self.weights)
    return self.aggregate(agg, arg_dict, **kwargs)

  @poller
  def vote(self, arg_dict: Dict = dict(), **kwargs) -> np.float:
    return self.aggregate(np.bincount, arg_dict, **kwargs).argmax()
