/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lorenzo_bbgky.h"

int
dsdgdt (double *s, double *hopmat, double *jvec, double *hvec, double drv,
	int latsize, double norm, double *dsdt)
{
  //Prepare the Levi civita symbol
  int *eps;
  eps = (int *) calloc (27, sizeof (int));
  //Allocate by the [x,y,z] element as eps[x + (y*3) + (z*9)]
  eps[0 + (1 * 3) + (2 * 9)] = 1;
  eps[1 + (2 * 3) + (0 * 9)] = 1;
  eps[2 + (0 * 3) + (1 * 9)] = 1;
  eps[0 + (2 * 3) + (1 * 9)] = -1;
  eps[2 + (1 * 3) + (0 * 9)] = -1;
  eps[1 + (0 * 3) + (2 * 9)] = -1;

  //Pointer to cmat
  double *cmat, *dcdt_mat;
  int m, n, b, g;		//xyz indices
  int i, j;			//lattice indices
  double rhs;

  cmat = &s[3 * latsize - 1];
  dcdt_mat = &dsdt[3 * latsize - 1];

  //Calculate the mean field contributions:
  //mf_s^\alpha_i =  \sum_k s^\alpha_k * hopmat_{ki}
  //mf_cmat^{\gamma + 3\beta}_{j+N*i} = \sum_k hopmat_{ik} * cmat^{\gamma+3\beta}_{i+N*k}
  double *mf_s = malloc (3 * latsize * sizeof (double));
  double *mf_cmat = malloc (9 * latsize * latsize * sizeof (double));

  cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0, hopmat,
	       latsize, s, 3, 0.0, mf_s, 3);

  for (b = 0; b < 3; b++)
    for (g = 0; g < 3; g++)
      {

	cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
		     1.0, hopmat, latsize,
		     &cmat[(g + 3 * b) * latsize * latsize], latsize, 0.0,
		     &mf_cmat[(g + 3 * b) * latsize * latsize], latsize);
      }

  //Update the spins in dsdt
  for (m = 0; m < 3; m++)
    for (i = 0; i < latsize; i++)
      {
	rhs = 0.0;
	for (b = 0; b < 3; b++)
	  for (g = 0; g < 3; g++)
	    {
	      rhs -= (hvec[b] + mf_s[i + 3 * b]) * s[i + 3 * g];
	      rhs -=
		mf_cmat[(g + 3 * b) * (latsize * latsize) +
			(i + latsize * i)];
	      rhs *= eps[m + (b * 3) + (g * 9)];	//[m,b,g]th element of levi civita
	    }
	dsdt[i + 3 * m] = 2.0 * rhs;
      }

  //Update the correlations in dgdt
  for (n = 0; n < 3; n++)
    for (m = n; m < 3; m++)
      for (i = 0; i < latsize; i++)
	for (j = 0; j < latsize; j++)
	  {
	    rhs = 0.0;
	    for (b = 0; b < 3; b++)
	      {
		rhs -=
		  hopmat[j + latsize * i] * (s[i * 3 * b] - s[j * 3 * b]);
		rhs *= eps[m + (n * 3) + (b * 9)];	//[m,n,b]th element of levi civita
	      }
	    for (b = 0; b < 3; b++)
	      for (g = 0; g < 3; g++)
		{
		  rhs -=
		    (hvec[b] + mf_s[i + 3 * b] -
		     hopmat[j + latsize * i] * s[j +
						 3 * b]) * cmat[((n +
								  3 * g) *
								 latsize *
								 latsize) +
								(j +
								 latsize *
								 i)] * eps[b +
									   (g
									    *
									    3)
									   +
									   (m
									    *
									    9)];
		  rhs -=
		    (hvec[b] + mf_s[j + 3 * b] -
		     hopmat[i + latsize * j] * s[i +
						 3 * b]) * cmat[((g +
								  3 * m) *
								 latsize *
								 latsize) +
								(j +
								 latsize *
								 i)] * eps[b +
									   (g
									    *
									    3)
									   +
									   (n
									    *
									    9)];
		  //Add the terms on Eq (B.4b), page 8 of manuscript and ur done!
		}
	    dcdt_mat[((n + 3 * m) * latsize * latsize) + (j + latsize * i)] =
	      rhs;
	    dcdt_mat[((m + 3 * n) * latsize * latsize) + (i + latsize * j)] =
	      rhs;
	  }

  free (eps);
  free (mf_s);
  free (mf_cmat);
  return 0;
}
