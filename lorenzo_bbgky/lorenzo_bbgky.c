/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lorenzo_bbgky.h"

//Levi civita symbol
int
eps (int i, int j, int k)
{
  int result = 0;
  if ((i == 0) && (j == 1) && (k == 2))
    {
      result = 1;
    }
  if ((i == 1) && (j == 2) && (k == 0))
    {
      result = 1;
    }
  if ((i == 2) && (j == 0) && (k == 1))
    {
      result = 1;
    }

  if ((i == 2) && (j == 1) && (k == 0))
    {
      result = -1;
    }
  if ((i == 1) && (j == 0) && (k == 2))
    {
      result = -1;
    }
  if ((i == 0) && (j == 2) && (k == 1))
    {
      result = -1;
    }

  return result;
}


int
dsdgdt (double *wspace, double *s, double *hopmat, double *jvec, double *hvec,
	int latsize, double norm, double *dsdt)
{
  double ncmprtol = 1e-10;	//Tolerance for comparing norm to unity
  //Pointer to cmat
  double *cmat, *dcdt_mat;
  int m, n, b, g;		//xyz indices
  int i, j;			//lattice indices
  double rhs, lastterm1, lastterm2;

  cmat = &s[3 * latsize];
  dcdt_mat = &dsdt[3 * latsize];

  //Normalize
  if (fabs (norm - 1.0) < ncmprtol)
    {
      for (i = 0; i < latsize; i++)
	for (j = i + 1; j < latsize; j++)
	  {
	    hopmat[i + latsize * j] = hopmat[i + latsize * j] / norm;
	    hopmat[j + latsize * i] = hopmat[i + latsize * j];
	  }
    }

  //Calculate the mean field contributions:
  //mf_s^\alpha_i =  \sum_k s^\alpha_k * hopmat_{ki}
  //mf_cmat^{\gamma + 3\beta}_{j+N*i} = \sum_k hopmat_{ik} * cmat^{\gamma+3\beta}_{i+N*k}
  double *mf_s, *mf_cmat;

  mf_s = &wspace[0];
  mf_cmat = &wspace[3 * latsize];

  cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0, hopmat,
	       latsize, s, latsize, 0.0, mf_s, latsize);

  for (b = 0; b < 3; b++)
    for (g = 0; g < 3; g++)
      {

	cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
		     1.0, hopmat, latsize,
		     &cmat[(g + 3 * b) * latsize * latsize], latsize, 0.0,
		     &mf_cmat[(g + 3 * b) * latsize * latsize], latsize);
      }

  //Update the spins in dsdt
  for (i = 0; i < latsize; i++)
    for (m = 0; m < 3; m++)
      {
	rhs = 0.0;
	for (g = 0; g < 3; g++)
	  {
	    for (b = 0; b < 3; b++)
	      {
		rhs -= hvec[b] * s[i + latsize * g] * eps (m, b, g);
		rhs -=
		  (mf_s[i + latsize * b] * s[i + latsize * g] +
		   mf_cmat[((g + 3 * b) * latsize * latsize) +
			   (i + latsize * i)]) * eps (m, b, g) * jvec[b];
	      }
	  }
	dsdt[i + latsize * m] = 2.0 * rhs;
      }

  //Update the correlations in dgdt DEBUG THIS ONLY!!!
  for (m = 0; m < 3; m++)
    for (n = m; n < 3; n++)
      for (i = 0; i < latsize; i++)
	for (j = 0; j < latsize; j++)
	  {
	    if (i != j)
	      {
		rhs = 0.0;
		for (b = 0; b < 3; b++)
		  {
		    rhs -=
		      (jvec[n] * s[i + latsize * b] -
		       jvec[m] * s[j + latsize * b]) * hopmat[j +
							      latsize * i] *
		      eps (m, n, b);
		  }
		for (b = 0; b < 3; b++)
		  for (g = 0; g < 3; g++)
		    {
		      rhs -=
			(hvec[b] + jvec[b] * (mf_s[i + latsize * b] -
					      hopmat[j + latsize * i] * s[j +
									  latsize
									  *
									  b]))
			* cmat[((n + 3 * g) * latsize * latsize) +
			       (j + latsize * i)] * eps (b, g,
							 m) + (hvec[b] +
							       jvec[b] *
							       (mf_s
								[j +
								 latsize *
								 b] -
								hopmat[i +
								       latsize
								       * j] *
								s[i +
								  latsize *
								  b])) *
			cmat[((g + 3 * m) * latsize * latsize) +
			     (j + latsize * i)] * eps (b, g, n);

		      rhs -= mf_cmat
			[((n + 3 * b) * latsize * latsize) +
			 (j + latsize * i)] * s[i +
						latsize * g] * eps (b,
								    g,
								    m)
			* jvec[b] +
			mf_cmat
			[((m + 3 * b) * latsize * latsize) +
			 (j + latsize * i)] * s[j + latsize * g] * eps (b, g,
									n) *
			jvec[b];

		      //Last term in the rhs of eqs (B.4b) in arXiv:1510.03768 
		      lastterm1 =
			cmat[((g + 3 * b) * latsize * latsize) +
			     (j + latsize * i)] + s[i + 3 * b] * s[j + 3 * g];
		      lastterm1 *= s[i + latsize * m] * eps (b, g, n);

		      lastterm2 =
			cmat[((b + 3 * g) * latsize * latsize) +
			     (j + latsize * i)] + s[i + 3 * g] * s[j + 3 * b];
		      lastterm2 *= s[i + latsize * n] * eps (b, g, m);

		      rhs +=
			(lastterm1 + lastterm2) * hopmat[j +
							 latsize * i] *
			jvec[b];
		    }
		dcdt_mat[((n + 3 * m) * latsize * latsize) +
			 (j + latsize * i)] = 2.0 * rhs;
		dcdt_mat[((m + 3 * n) * latsize * latsize) +
			 (i + latsize * j)] = 2.0 * rhs;
	      }
	    else
	      {
		dcdt_mat[((n + 3 * m) * latsize * latsize) +
			 (j + latsize * i)] = 0.0;
	      }
	  }

  return 0;
}
