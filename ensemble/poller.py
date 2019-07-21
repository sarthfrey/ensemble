from functools import wraps
from .types import PollingStrategy


def poller(function):
  @wraps(function)
  def wrapper(self, *args, **kwargs):
    if 'arg_dict' in kwargs:
      arg_dict = kwargs.get('arg_dict')
      kwargs.pop('arg_dict', None)
    elif len(args) > 0:
      arg_dict = args[-1]
    else:
      arg_dict = dict()
    if not isinstance(arg_dict, dict):
      raise ValueError('arg_dict must be a dict, you may not specify other positional args to Ensemble')
    if arg_dict and set(arg_dict.keys()) != set(self.children):
      raise ValueError('The keys of arg_dict must be the same as the names of the children of the ensemble')
    if arg_dict:
      self.set_polling_strategy('structured')
    else:
      self.set_polling_strategy('flat')
    return function(self, *args, **kwargs)
  return wrapper
