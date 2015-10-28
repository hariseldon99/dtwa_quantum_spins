/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lorenzo_bbgky.h"

int
dsdgdt (double *s, double *hopmat, double *jvec, double *hvec, double drv,
	int latsize, double norm, double *dsdt)
{
  //Prepare the Levi civita symbol DEBUG THIS!!! CONVERT TO ndarr
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
  double rhs, lastterm1, lastterm2;

  cmat = &s[3 * latsize];
  dcdt_mat = &dsdt[3 * latsize];

  //Calculate the mean field contributions:
  //mf_s^\alpha_i =  \sum_k s^\alpha_k * hopmat_{ki}
  //mf_cmat^{\gamma + 3\beta}_{j+N*i} = \sum_k hopmat_{ik} * cmat^{\gamma+3\beta}_{i+N*k}
  double *mf_s = malloc (3 * latsize * sizeof (double));
  double *mf_cmat = malloc (9 * latsize * latsize * sizeof (double));

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
  for (m = 0; m < 3; m++)
    for (i = 0; i < latsize; i++)
      {
	rhs = 0.0;
	for (b = 0; b < 3; b++)
	  for (g = 0; g < 3; g++)
	    {
	      rhs -= (hvec[b] + mf_s[i + latsize * b]) * s[i + latsize * g];
	      rhs -=
		mf_cmat[(g + 3 * b) * (latsize * latsize) +
			(i + latsize * i)];
	      rhs *= eps[m + (b * 3) + (g * 9)];	//[m,b,g]th element of levi civita
	    }
	dsdt[i + latsize * m] = 2.0 * rhs;
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
		  hopmat[j + latsize * i] * (s[i + latsize * b] -
					     s[j + latsize * b]);
		rhs *= eps[m + (n * 3) + (b * 9)];	//[m,n,b]th element of levi civita
	      }
	    for (b = 0; b < 3; b++)
	      for (g = 0; g < 3; g++)
		{
		  rhs -=
		    (hvec[b] + mf_s[i + latsize * b] -
		     hopmat[j + latsize * i] * s[j +
						 latsize * b]) * cmat[((n +
									3 *
									g) *
								       latsize
								       *
								       latsize)
								      + (j +
									 latsize
									 *
									 i)] *
		    eps[b + (g * 3) + (m * 9)];
		  rhs -=
		    (hvec[b] + mf_s[j + latsize * b] -
		     hopmat[i + latsize * j] * s[i +
						 latsize * b]) * cmat[((g +
									3 *
									m) *
								       latsize
								       *
								       latsize)
								      + (j +
									 latsize
									 *
									 i)] *
		    eps[b + (g * 3) + (n * 9)];
		  //RHS of Eq (B.4b), page 8 of arXiv:1510.03768 
		  rhs -=
		    (mf_cmat
		     [((n + 3 * b) * latsize * latsize) + (j + latsize * i)] -
		     hopmat[j +
			    latsize * i] * cmat[((n + 3 * b) * latsize *
						 latsize) + (j +
							     latsize * j)]) *
		    s[i + latsize * g] * eps[b + (g * 3) + (m * 9)];
		  rhs -=
		    (mf_cmat
		     [((m + 3 * b) * latsize * latsize) + (j + latsize * i)] -
		     hopmat[i +
			    latsize * j] * cmat[((m + 3 * b) * latsize *
						 latsize) + (i +
							     latsize * i)]) *
		    s[j + latsize * g] * eps[b + (g * 3) + (n * 9)];

		  //Last term in the rhs of eqs (B.4b) in arXiv:1510.03768 
		  lastterm1 =
		    cmat[((g + 3 * b) * latsize * latsize) +
			 (j + latsize * i)] + s[i + 3 * b] * s[j + 3 * g];
		  lastterm1 *=
		    s[i + latsize * m] * eps[b + (g * 3) + (n * 9)];

		  lastterm2 =
		    cmat[((b + 3 * g) * latsize * latsize) +
			 (j + latsize * i)] + s[i + 3 * g] * s[j + 3 * b];
		  lastterm2 *=
		    s[i + latsize * n] * eps[b + (g * 3) + (m * 9)];

		  rhs += (lastterm1 + lastterm2) * hopmat[j + latsize * i];
		}
	    dcdt_mat[((n + 3 * m) * latsize * latsize) + (j + latsize * i)] =
	      2.0 * rhs;
	    dcdt_mat[((m + 3 * n) * latsize * latsize) + (i + latsize * j)] =
	      2.0 * rhs;
	  }

  free (eps);
  free (mf_s);
  free (mf_cmat);
  return 0;
}