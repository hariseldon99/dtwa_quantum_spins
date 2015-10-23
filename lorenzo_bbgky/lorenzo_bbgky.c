/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lorenzo_bbgky.h"

int
dsdg (double *s, double *hopmat, double *jvec, double *hvec, double drv,
      double latsize, double norm, double *dsdt)
{
  int outcome = GSL_SUCCESS;

  size_t svecsize = 3 * latsize;
  size_t fullsize = svecsize + 9 * latsize * latsize;
  gsl_vector_view c_gplus3b;
  gsl_matrix_view c_gplus3b_mat;

  gsl_matrix_view jmat = gsl_matrix_view_array (hopmat, latsize, latsize);
  //Put the 3 s vectors  a gsl_matrix 
  gsl_matrix_view amat = gsl_matrix_view_array (s, 3, latsize);
  gsl_matrix_view dadt_mat = gsl_matrix_view_array (dsdt, 3, latsize);

  //Use matrix views to reshape the rest of the elements into a [3*3]X[L*L] matrix
  int cmat_dataptr = (int *) (s + (svecsize - 1));
  int dcdt_mat_dataptr = (int *) (dsdt + (svecsize - 1));

  gsl_matrix_view cmat =
    gsl_matrix_view_array (cmat_dataptr, 9, latsize * latsize);

  gsl_matrix_view dcdt_mat =
    gsl_matrix_view_array (dcdt_mat_dataptr, 9, latsize * latsize);

  //Do the mean field contributions in cblas 
  gsl_matrix *smat_mf = gsl_matrix_alloc (3, latsize);
  gsl_matrix *cmat_mf = gsl_matrix_alloc (9, latsize * latsize);

  gsl_vector_view cmat_mf_gplus3b;
  gsl_matrix_view cmat_mf_gplus3b_mat;

  outcome = gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
			    1.0, &amat.matrix, &jmat.matrix,
			    0.0, smat_mf);
  int g, b;

  for (g = 0; g < 3; g++)
    {
      for (b = 0; b < 3; b++)
	{
	  //For each gamma, beta, get the (gamma+ 3*beta)^th row of cmat
	  c_gplus3b = gsl_matrix_row (&cmat.matrix, g + 3 * b);
	  cmat_mf_gplus3b = gsl_matrix_row (cmat_mf, g + 3 * b);

	  //Reshape view to a matrix
	  c_gplus3b_mat =
	    gsl_matrix_view_vector (&c_gplus3b.vector, latsize, latsize);
	  cmat_mf_gplus3b_mat =
	    gsl_matrix_view_vector (&cmat_mf_gplus3b.vector, latsize, latsize);

	  //Compute the matrix product J * cmat
	  outcome = gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
				    1.0, &jmat.matrix, &c_gplus3b_mat.matrix,
				    0.0, &cmat_mf_gplus3b_mat.matrix);
	}
    }


  //Loop over all other unique indices to get the rhs

  //Free the mean field matrices
  gsl_matrix_free (smat_mf);
  gsl_matrix_free (cmat_mf);

  return outcome;

}
