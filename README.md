# ensemble

Model ensemble utility optimized for low barrier integration. If you find yourself needing to use one function to call many functions, this is what you need.

### Examples

You can use `Ensemble` to multiplex your functions

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


