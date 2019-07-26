from ensemble import Ensemble
from pprint import pprint


def model1(x):
  return x % 3 == 0

def model2(x):
  return x % 5 == 0

def get_dataset():
  return [(i, i % 15 == 0) for i in range(1, 101)]

def get_results(dataset, preds):
  labels = [label for _, label in dataset]
  positives = sum(1 for label in labels if label)
  predicted_positives = sum(1 for pred in preds if pred)
  true_positives = sum(1 for label, pred in zip(labels, preds) if label and pred)
  return 100.0 * true_positives / predicted_positives, 100.0 * true_positives / positives

def evaluate(model):
  def wrapper(dataset):
    preds = [model(x=x) for x, _ in dataset]
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
  print(results)
  pprint(
    results(dataset=get_dataset())
  )

"""
{'ensemble': {'precision': '100.0%', 'recall': '100.0%'},
 'model1': {'precision': '18.2%', 'recall': '100.0%'},
 'model2': {'precision': '30.0%', 'recall': '100.0%'}}
"""

