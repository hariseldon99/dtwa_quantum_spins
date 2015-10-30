#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <cblas.h>

/* Functions */
int eps (int i, int j, int k);

int dsdgdt (double *, double *, double *, double *, double *, int,
	    double, double *);
