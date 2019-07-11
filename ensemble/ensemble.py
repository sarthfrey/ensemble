import pprint
import json
from collections import defaultdict


class Ensemble(object):
  model_functions = {}
  ensemble_groups = defaultdict(set)

  def __init__(self, name, model_fns=[]):
    self.name = name
    if not name or not isinstance(name, str):
      raise ValueError('Ensemble name must be a non-empty string')
    for model_function in model_fns:
      self.model_functions[model_function.__name__] = model_function
      self.ensemble_groups[model_function.__name__] |= set([self.name])

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
    m = ''.join(f'    \'{k}\': {pprint.pformat(v)}\n' for k, v in self.get_models_functions())
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

  def get_models_functions(self):
    return (
      (n, m) for n, m in self.model_functions.items() if self.name in self.ensemble_groups[n]
    )

  def call_all_models(self, *args, **kwargs):
    return {
      n: m(*args, **kwargs) for n, m in self.get_models_functions()
    }
