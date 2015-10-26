/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lorenzo_bbgky.h"

int
dsdg (double *s, double *hopmat, double *jvec, double *hvec, double drv,
      int latsize, double norm, double *dsdt)
{
  //Pointer to cmat
  double *cmat ;
  
  cmat = &s[3*latsize-1];
  
  //Calculate the mean field contributions \sum_k s^\alpha_k * hopmat_{ki}
  double *mf_s = malloc(3*latsize*sizeof(double));
  double *mf_cmat = malloc(9*latsize*latsize*sizeof(double));
  
  cblas_dsymm(CblasRowMajor,CblasRight,CblasUpper,3,latsize,1.0, hopmat, latsize,s,3,0.0,mf_s,3);

  free (mf_s);
  free (mf_cmat);
  return 0;

}
