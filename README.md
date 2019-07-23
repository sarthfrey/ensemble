# ensemble

*A model ensemble package optimized for low barrier integration*

![Model Ensemble](img/main.png)

**ensemble** lets you combine your models and access them by a single object. You may use that ensemble to multiplex between your models, call them all, and aggregate the results. You may even create ensembles of ensembles!

This package borrows the idea of computation graph sessioning from [TensorFlow](https://github.com/tensorflow/tensorflow) and implements the [composite pattern](https://en.wikipedia.org/wiki/Composite_pattern) for building tree hierarchies.

### Documentation

[![Documentation Status](https://readthedocs.org/projects/ensemble-pkg/badge/?version=latest)](https://ensemble-pkg.readthedocs.io/en/latest/?badge=latest)

Read the docs at [ensemble-pkg.readthedocs.io](https://ensemble-pkg.readthedocs.io)

### Installation

```
pip install ensemble-pkg
```

### Case Study

Let's say we have two models that accomplish a binary classification task. We want to be able to easily combine the two models into an ensemble and then test the precision and recall of both models and the ensemble. With this package, it's easy.

We start off by building our two models for a dataset. In this case, the task is classifying if a number is divisble by 15. We have two models, one says the number is divisble by 15 if it is divisble by 3 and the other says so if it is divisble by 5.

```python
def model1(x):
  return x % 3 == 0

def model2(x):
  return x % 5 == 0

def get_dataset():
  return [(i, i % 15 == 0) for i in range(1, 101)]
```

Define a function that gets our precision and recall for a given dataset and set of predictions:

```python
def get_results(dataset, preds):
  labels = [label for _, label in dataset]
  positives = sum(1 for label in labels if label)
  predicted_positives = sum(1 for pred in preds if pred)
  true_positives = sum(1 for label, pred in zip(labels, preds) if label and pred)
  return 100.0 * true_positives / predicted_positives, 100.0 * true_positives / positives
```

Next we build a model ensemble from `model1` and `model2`, specifically one that only outputs `True` if all its children output `True`. In this case, the ensemble would then only output `True` if the input is both divisible by 3 and 5.

```python
e = Ensemble('ensemble', children=[model1, model2], mode='all')
e(x=3) # returns False
e(x=5) # returns False
e(x=15) # returns True
```

Finally, lets build another ensemble from our two models and the ensemble in order to easily aggregate the precision and recall stats for each model. We do this by decorating each child model with an evaluation decorator which modifies the models to take a dataset and output precision and recall, instead of taking a number and outputing `True` or `False`.

```python
def evaluate(model):
  def wrapper(dataset):
    preds = [model(x=x) for x, _ in dataset]
    precision, recall = get_results(dataset, preds)
    return {
      'precision': f'{precision:.1f}%',
      'recall': f'{recall:.1f}%',
    }
  return wrapper

results = Ensemble('results', children=[model1, model2, e])
results.decorate_children(evaluate)
```

Finally we run `results(dataset=get_dataset())` and get the following results as expected!

```
{'ensemble': {'precision': '100.0%', 'recall': '100.0%'},
 'model1': {'precision': '18.2%', 'recall': '100.0%'},
 'model2': {'precision': '30.0%', 'recall': '100.0%'}}
```

In a few keystrokes, we built the following graph structure.

<img alt="Graph Structure" src="img/graph.png" width="20%" height="20%"/>

### Examples

Define your model functions and create your ensemble:

```python
>>> from ensemble import Ensemble
>>> def square(x):
...     return x**2
>>> def cube(x):
...     return x**3
>>> e = Ensemble(name='e1', children=[square, cube])
```

Call all the models in the ensemble:
```python
>>> e(x=2)
{'square': 4, 'cube': 8}
>>> e(x=3)
{'square': 9, 'cube': 27}
```

Multiplex between functions:

```python
>>> e.multiplex('square', x=2)
4
>>> e.multiplex('cube', x=3)
27
```

You may instead decorate your model functions with `@child` in order to attach them to an ensemble:

```python
>>> from ensemble import child
>>> @child('e2')
... def func1(x):
...     return x**2
...
>>> @child('e2')
... def func2(x):
...     return x**3
...
>>> e = Ensemble('e2')
>>> e(x=3)
{'func1': 9, 'func2': 27}
```

You may even attach a model to multiple ensembles!

```python
>>> @child('e2', 'e3')
... def func3(x, y):
...     return x**3 + y
...
>>> e2(x=2, y=3)
{'func1': 4, 'func2': 8, 'func3': 11}
>>>
>>> e3 = Ensemble('e3')
>>> e3(x=2, y=3)
{'func3': 11}
```

Compute statstical aggregations from your ensemble's models:

```python
>>> def a(x):
...   return x + 1
...
>>> def b(y):
...   return y + 2
...
>>> def c(z):
...   return z + 2
...
>>> e = Ensemble('e4', children=[a, b], weights=[3.0, 1.0])
>>> e.mean(x=2, y=3)
4.0
>>> e.weighted_mean(x=2, y=3)
3.5
>>> e.weighted_sum(x=2, y=3)
14.0
>>> e = Ensemble('e6', [a, b, c])
>>> e.vote(x=1, y=1, z=1)
3
```

Build ensembles of ensembles!

```python
>>> first_ensemble = Ensemble('first', children=[c])
>>> second_ensemble = Ensemble('second', children=[a, b])
>>> parent_ensemble = Ensemble('parent', children=[first_ensemble, second_ensemble])
>>> parent_ensemble(x=1, y=1, z=1)
{'first': {'c': 3}, 'second': {'a': 2, 'b': 3}}
>>> parent_ensemble.multiplex('second', x=3, y=1)
{'a': 4, 'b': 3}
```

Use that idea to chain aggregate computations! Compute the mean of the sum of the model outputs in each ensemble:

```python
>>> first_ensemble.set_mode('sum')
Ensemble(name='first', children=['c'], weights=None, mode='sum')
>>> second_ensemble.set_mode('sum')
Ensemble(name='second', children=['a', 'b'], weights=None, mode='sum')
>>> parent_ensemble.mean(x=1, y=1, z=1)
4.0
```

If you forget what models are in your ensemble, just check:

```python
>>> print(parent_ensemble)
Ensemble(name='parent', children=['first', 'second'], weights=None, mode='all')
  Ensemble(name='first', children=['c'], weights=None, mode='sum')
    Model(name='c', func=c(z))
  Ensemble(name='second', children=['a', 'b'], weights=None, mode='sum')
    Model(name='a', func=a(x))
    Model(name='b', func=b(y))
```

In the above example, a tree is shown which shows which models and ensembles are the children of which ensembles!
