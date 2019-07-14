ensemble
========

ensemble will solve your problem of where to start with documentation,
by providing a basic explanation of how to do it easily.

Look how easy it is to use:

    from ensemble import Ensemble

    # define a model
    def square(x):
      return x**2

    # build an ensemble
    e = Ensemble(name='e1', children=[square])

    # call square from the ensemble
    e.multiplex('square', x=2) # returns 4

Features
--------

- Create model ensembles
- Multiplex between models
- Call all models
- Aggregate model results
- Do weighted sums, means, votes, and more

Installation
------------

Install ensemble by running:

    pip install ensemble-pkg

Contribute
----------

- Issue Tracker: https://github.com/sarthfrey/ensemble/issues
- Source Code: https://github.com/sarthfrey/ensemble

Support
-------

If you are having issues, please let us know! Submit an issue, 
or if you're feeling adventurous, a PR :)

License
-------

The project is licensed under the MIT license.