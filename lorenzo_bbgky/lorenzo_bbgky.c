/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lorenzo_bbgky.h"

int
dsdg (double *s, double **hopmat, double **jvec, double **hvec, double drv, double latsize, double norm , double *dsdt)
{
  //Put s into a gsl_vector and use vector views to reshape the first 3L elements into a matrix
  //Use matrix views to reshape the rest of the elements into a [3][3][L][L] tensor with the first two and last two indices
  //flattened
  //Do the mean field contributions in cblas 
  //Loop over all other unique indices to get the rhs
  
  return GSL_SUCCESS;

}
