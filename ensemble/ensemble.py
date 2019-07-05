from collections import defaultdict


class Ensemble(object):
  model_functions = {}
  ensemble_groups = defaultdict(set)
  ensemble = ''

  def __init__(self, specifier, *ensemble_names):
    model_function = None
    if callable(specifier):
      model_function = specifier
      ensemble_names = set(ensemble_names)
    else:
      ensemble_names = set([specifier])
    if any(not ensemble_name for ensemble_name in ensemble_names):
      raise ValueError('Must provide a valid ensemble names')
    if model_function is None:
      self.ensemble = specifier
    else:
      self.model_function = model_function
      self.ensemble_groups[model_function.__name__] |= ensemble_names
      self.model_functions[model_function.__name__] = model_function

  def __call__(self, *args, **kwargs):
    if 'model' not in kwargs:
      return self.model_function(*args, **kwargs)
    model = kwargs.get('model')
    if model not in self.model_functions or model not in self.ensemble_groups:
      raise ValueError('There is no decorated model function `{}`'.format(model))
    model_function = self.model_functions[model]
    ensemble_group = self.ensemble_groups[model]
    if self.ensemble not in ensemble_group:
      raise ValueError('Model function `{}` is not attached to ensemble `{}`'.format(model, self.ensemble))
    kwargs.pop('model', None)
    return model_function(*args, **kwargs)


class _Model(object):
  def __init__(self, *ensemble_names):
    self.ensemble_names = ensemble_names

  def __call__(self, model_function):
    return Ensemble(model_function, *self.ensemble_names)

def model(*ensemble_names):
  return _Model(*ensemble_names)
