
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

In a `companion preprint article <https://arxiv.org/abs/2005.03220>`_, we show
that the proposed method is fast and scalable for large-scale data problems, and
delivers results that are straightforward to interpret and compare across models
and datasets.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   auto_examples/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
