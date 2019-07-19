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
  DEFAULT_MODE = 'all'