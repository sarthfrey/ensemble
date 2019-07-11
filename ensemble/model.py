import inspect

from .ensemble import Ensemble


class _Model(Ensemble):
  invalid_args_names = [
    'model',
  ]

  def __init__(self, model_function, *ensemble_names):
    self.model_function = model_function
    self.ensemble_names = set(ensemble_names)
    if any(not ensemble_name for ensemble_name in self.ensemble_names):
      raise ValueError('Must provide a valid ensemble names')
    super().model_functions[model_function.__name__] = model_function
    super().ensemble_groups[model_function.__name__] |= self.ensemble_names
    _Model._validate_model_function(model_function)

  def __call__(self, *args, **kwargs):
    return self.model_function(*args, **kwargs)

  @staticmethod
  def _validate_model_function(model_function):
    arg_names = inspect.getfullargspec(model_function)[0]
    for invalid_arg_name in _Model.invalid_args_names:
      if invalid_arg_name in arg_names:
        raise ValueError(
          f'Function `{model_function.__name__}` is decorated with @model '
          f'and so it may not have `{invalid_arg_name}` as an argument'
        )

def model(*ensemble_names):
  def wrapper(model_function):
    return _Model(model_function, *ensemble_names)
  return wrapper
