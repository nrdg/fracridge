
`fracridge` : fractional ridge regression
=========================================

Ridge regression (RR) is a regularization technique that penalizes the L2-norm
of the coefficients in linear regression. One of the challenges of using RR is
the need to set a hyperparameter (α) that controls the amount of regularization.
Cross-validation is typically used to select the best α from a set of
candidates. However, efficient and appropriate selection of α can be
challenging, particularly where large amounts of data are analyzed. Because the
selected α depends on the scale of the data and predictors, it is also not
straightforwardly interpretable.

Here, we reparameterize RR in terms of the ratio γ between the L2-norms of the
regularized and unregularized coefficients.

This approach, called fractional RR (FRR), has several benefits:
the solutions obtained for different γ are guaranteed to vary, guarding against
wasted calculations, and automatically span the relevant range of
regularization, avoiding the need for arduous manual exploration.

In a `companion article <https://academic.oup.com/gigascience/article/9/12/giaa133/6011381>`_, we show that the proposed method is fast and scalable for large-scale data problems, and delivers results that are straightforward to interpret and compare across models and datasets.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/lc_LLF-iA_c" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   contributing
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

`Getting started <quick_start.html>`_
-------------------------------------

Information regarding installation and basic usage.

`User Guide <user_guide.html>`_
-------------------------------

How to use the software.

`Contributing to Fracridge development <contributing.html>`_
-------------------------------------------------------------

How to report issue and contribute enhancements to the software.

`API Documentation <api.html>`_
-------------------------------

An example of API documentation.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples. It complements the `User Guide <user_guide.html>`_.
