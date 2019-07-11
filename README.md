# ensemble

*A model ensemble utility optimized for low barrier integration*

**TL;DR** if you find yourself needing to use one function to call many functions, this is what you need.

### Examples

Define your model functions and create your ensemble:

```
>>> from ensemble import Ensemble
>>> def function1(x):
...     return x**2
...
>>> def function2(y):
...     return y**3
...
>>> my_ensemble = Ensemble(
...     name='e1',
...     model_fns=[function1, function2],
... )
```

Multiplex between functions:

```
>>> my_ensemble(model='function1', x=2)
4
>>> my_ensemble(model='function2', y=2)
8
```

Call all the models in the ensemble:

```
>>> my_ensemble.all(x=4, y=3)
{'function1': 16, 'function2': 27}
```

You may instead decorate your model functions with `@model` in order to attach them to an ensemble:

```
>>> from ensemble import model
>>> @model('e2')
... def func1(x):
...     return x**2
...
>>> @model('e2')
... def func2(x):
...     return x**3
...
>>> e2 = Ensemble('e2')
>>> e2.all(x=3)
{'func1': 9, 'func2': 27}
```

You may even attach a model function to multiple ensembles!

```
>>> @model('e2', 'e3')
... def func3(x, y):
...     return x**3 + y
...
>>> e2.all(x=2, y=3)
{'func1': 4, 'func2': 8, 'func3': 11}
>>>
>>> e3 = Ensemble('e3')
>>> e3.all(x=2, y=3)
{'func3': 11}
```

If you forget what models are in your ensemble, just check!

```
>>> e2
Ensemble(
  name='e2',
  model_functions={
    'func1': <function func1 at 0x1024fa9d8>
    'func2': <function func2 at 0x1024faa60>
    'func3': <function func3 at 0x1024fa950>
  }
)
>>> e3
Ensemble(
  name='e3',
  model_functions={
    'func3': <function func3 at 0x1024fa950>
  }
)
```

In the above example, ensemble `e2` contains `func1`, `func2`, and `func3`, while ensemble `e3` contains just `func3`.
