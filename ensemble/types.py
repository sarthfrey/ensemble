from enum import Enum


class PollingStrategy(Enum):
  STRUCTURED = 'structured'
  FLAT = 'flat'


class CallMode(Enum):
  MULTIPLEX = 'multiplex'
  ALL = 'all'
  AGGREGATE = 'aggregate'
  SUM = 'sum'
  MEAN = 'mean'
  DEFAULT_MODE = 'call_children'
  MAX = 'max'
  ANY = 'any'
  CALL_CHILDREN = 'call_children'
  VOTE = 'vote'
