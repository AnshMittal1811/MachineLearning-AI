This mini library gathers functions for computing:

- Spherical harmonics in real or complex basis (up to degree 3)
- Wigner matrices (real or complex) of any degree
- Clebsch-Gordan coefficients and projections for decomposing tensor products
  of irreducible SO( 3 ) representations

The useful functions are commented in the python files here is a general description:

Let Y^l_k be the spherical harmonics basis for l in NN and k in [|-l, l|]
For any rotation matrix R in SO( 3 ) anx x in the unit sphere S_2 we have:
Y^l( R^{-1} x ) = D^l(R)Y^l where D^l(R) is the Wigner matrix.

For any p,q in NN and J in [||p-q|, p+q|]
we have Q^{p,q,J}D^p \otimes D^q (Q^{p,q,J})^* = D^J

- The spherical harmonics Y^l_k can be computed in numpy using np_spherical_harmonics.py
  or in tensorflow using tf_spherical_harmonics.py

- The Wigner matrices D^l can be computed in numpy using wigner_matrix.py

- Clebsch-Gordan coefficients and the matrices Q^{p,q,J} can be computed using clebsch_gordan_decomposition.py
  Use the npClebschGordanMatrices class to compute the Q^{p,q,J} matrices in numpy
  or tf_clebsch_gordan_matrices to compute them in tensorflow

- higher_clebsch_gordan.py allows to compute decomposition of tensors of higher order
  (see RotationInvariance.pdf for details)
  use npHigherClebschGordan class to compute higher Clebsch-Gordan projectors
  use npClebschGordanPolynomial to compute the expansion of coefficients of Clebsch-Gordan
  projections as polynomials (see comments and RotationInvariance.pdf for details)
  use npInvariantPolynomials to compute polynomials invariant under the Wigner action

Useful references:

https://en.wikipedia.org/wiki/Spherical_harmonics
https://en.wikipedia.org/wiki/Wigner_D-matrix
https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients