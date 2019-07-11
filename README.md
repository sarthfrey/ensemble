# ensemble

*A model ensemble utility optimized for low barrier integration*

**TL;DR** if you find yourself needing to use one function to call many functions, this is what you need.

### Examples

choose between functions

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
>>> my_ensemble(model='function1', x=2)
4
>>> my_ensemble(model='function2', y=2)
8
```

you may even call them all

```
>>> my_ensemble.all(x=4, y=3)
{'function1': 16, 'function2': 27}
```

if you have a monolithic codebase you may simply decorate your mode functions in order to attach them to an ensemble

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
>>> my_ensemble = Ensemble('e2')
>>> my_ensemble.all(x=3)
{'func1': 9, 'func2': 27}
```


