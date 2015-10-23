#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

/* Functions */
int dsdg (double *, double *, double *, double *, double, double, double,
	  double *);
