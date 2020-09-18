.. title:: User guide : contents

.. _user_guide:

==========
User guide
==========

MATLAB
------

The MATLAB implementation provides a single entry-point function called ``fracridge``.

To use it, arrange your data ``y``, to have d rows by t columns, where ``d``
represents the number of observations in each target, and ``t`` is the number of
targets (which may be 1). Correspondingly, your design matrix ``X`` should have
d rows and p columns, where ``p`` is the number of parameters. The fractions
``frac`` are the requested fractions of the L2-norm of the regularized solutions
relative to the unregularized solution.

For detailed documentation, see also `API documentation <api.html>`_


Python
------

The Python implementation can be used in two different ways. The first
is a

Frequently asked questions
===========================

How should I set the ``frac`` input
------------------------------------

The ``frac`` input determines the L2-norm of the regularized coefficients for
the linear regression problem, *relative* to the L2-norm of the unregularized
coefficients for the same problem. If you are not sure how to set this, choose
an equally spaced set of fractions between 0 and 1 (not including 0). This
should give you a range of solutions between highly-regularized (small values)
and completely unregularized (1). To choose the 'right' solution, you will want
to use a strategy such as cross-validation (see the
`Scikit Learn documentation <https://scikit-learn.org/stable/modules/cross_validation.html>`_)
for an explanation of this concept.

How do I interpret the alphas output?
-------------------------------------

The values of ``alpha`` depend on the scale.


How do I compare the best fractions across targets?
----------------------------------------------------
