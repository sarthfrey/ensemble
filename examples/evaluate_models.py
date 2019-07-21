from ensemble import Ensemble
from pprint import pprint


def model1(x):
  return x % 3 == 0

def model2(x):
  return x % 5 == 0

def get_dataset():
  return [i % 15 == 0 for i in range(1, 101)]

def get_results(dataset, preds):
  positives = sum(1 for x, y in zip(dataset, preds) if x)
  predicted_positives = sum(1 for x, y in zip(dataset, preds) if y)
  true_positives = sum(1 for x, y in zip(dataset, preds) if y and x == y)
  return 100.0 * true_positives / predicted_positives, 100.0 * true_positives / positives

def evaluate(model):
  def wrapper():
    dataset = get_dataset()
    preds = [model(x=x) for x in range(1, 101)]
    precision, recall = get_results(dataset, preds)
    return {
      'precision': f'{precision:.1f}%',
      'recall': f'{recall:.1f}%',
    }
  return wrapper

if __name__ == '__main__':
  e = Ensemble('ensemble', children=[model1, model2], mode='all')
  results = Ensemble('results', children=[model1, model2, e])
  results.decorate_children(evaluate)
  pprint(results())

"""
{'ensemble': {'precision': '100.0%', 'recall': '100.0%'},
 'model1': {'precision': '18.2%', 'recall': '100.0%'},
 'model2': {'precision': '30.0%', 'recall': '100.0%'}}
"""

