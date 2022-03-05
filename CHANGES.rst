1.4 (March 04, 2022)
====================
  * made coef.squeeze in fracridge less eager (#40)
  * added lapack import & updated too long string; fixes #33 (#34)
  * Optimizes the computation to use best option for order of operations. (#30)
  * MAINT: Support version of Python up to 3.9 (#29)
  * BF: Propagate keyword arguments in the CV estimator. (#26)
  * DOC: Explains how to upgrade the software to newer releases. (#27)
  * Single regressor (#25)


1.3.2 (February 3rd, 2021)
==========================
Fixes a bug in cases where the number of targets is larger than
the number of samples in each target.

1.3 (December 2, 2020)
=========================
This version includes changes to the documentation, to refer to the
published paper.

1.2.1 (September 22, 2020)
==========================
This micro version fixes a few small things in the documentation.

  * Doc fixes (#15)


1.2 (September 22, 2020)
========================
This version substantially overhauls the documentation and adds a
cross-validation estimator.

  * Documentation overhaul. (#12)
  * FracRidgeCV estimator (#13)


Version 1.1 (June 29, 2020)
===================

This version adds more documentation

Version 1.0 (May 07, 2020)
==========================

This is the first major release of the software, in tandem with the appearance
of the preprint about the method.
