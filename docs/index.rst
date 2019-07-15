.. ensemble documentation master file, created by
   sphinx-quickstart on Sun Jul 14 11:28:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ensemble's docs!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

ensemble lets you combine your models and access them by a single object, called an ensemble. You may use that ensemble to multiplex between your models, call them all, and aggregate the results. You may even create ensembles of ensembles. ensemble borrows the idea of computation graph sessioning from tensorflow and implements the composite pattern for building tree hierarchies.

Look how easy it is to use:

.. code-block:: python

    from ensemble import Ensemble

    # define a model
    def square(x):
      return x**2

    # build an ensemble
    e = Ensemble(name='e1', children=[square])

    # call square from the ensemble
    e.multiplex('square', x=2) # returns 4

Installation
------------

Install ensemble by running::

    pip install ensemble-pkg

Features
--------

- Create model ensembles
- Multiplex between models
- Call all models
- Aggregate model results
- Do weighted sums, means, votes, and more

The API Documentation / Guide
-----------------------------

If you are looking for information on a specific function, class, or method,
this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   api

Contribute
----------

- Issue Tracker: https://github.com/sarthfrey/ensemble/issues
- Source Code: https://github.com/sarthfrey/ensemble

Support
-------

If you are having issues, please let us know! Submit an issue, or if you're feeling adventurous, a PR :)

License
-------

The project is licensed under the MIT license.
