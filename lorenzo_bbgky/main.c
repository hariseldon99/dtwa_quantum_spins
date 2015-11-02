#include<stdio.h>
#include "lorenzo_bbgky.h"

#define R 3
#define C 2
#define N 6			//Make sure that N = R * C

#define JX 1.0
#define JY 2.0
#define JZ 3.0
#define HX 3.0
#define HY 2.0
#define HZ 1.0
#define ALPHA 1.0

int
main (void)
{
  int mu, nu;
  int mux, nux, muy, nuy;
  double jmat[N * N];
  char delim;

  double dmn;
  double *workspace, *s, *dsdt;

  double jvec[3] = { JX, JY, JZ };
  double hvec[3] = { HX, HY, HZ };

  workspace = (double *) malloc ((3 * N + 9 * N * N) * sizeof (double));
  s = (double *) calloc ((3 * N + 9 * N * N), sizeof (double));
  dsdt = (double *) malloc ((3 * N + 9 * N * N) * sizeof (double));

  //sx are all 1
  for (mu = 0; mu < N; mu++)
    {
      s[mu] = 1.0;
    }

  //sy are 1, -1 alternates
  for (mu = N; mu < 2 * N; mu += 2)
    {
      s[mu] = 1.0;
      s[mu + 1] = -1.0;
    }
  //sz are 1, -1 alternates
  for (mu = 2 * N; mu < 3 * N; mu += 2)
    {
      s[mu] = 1.0;
      s[mu + 1] = -1.0;
    }

  for (mu = 0; mu < N; mu++)
    for (nu = mu; nu < N; nu++)
      {
	if (mu == nu)
	  {
	    jmat[mu + N * nu] = 0.0;
	  }
	else
	  {
	    mux = nu % C, nux = mu % C;
	    muy = nu % R, nuy = mu % R;
	    dmn =
	      sqrt ((mux - nux) * (mux - nux) + (muy - nuy) * (muy - nuy));
	    jmat[mu + N * nu] = 1.0 / pow (dmn, ALPHA);
	    jmat[nu + N * mu] = jmat[mu + N * nu];
	  }
      }

  printf ("\n\nHopping Matrix:");
  for (mu = 0; mu < N; mu++)
    {
      printf ("\n[");
      for (nu = 0; nu < N; nu++)
	{
	  if (nu == N - 1)
	    delim = ']';
	  else
	    delim = ',';
	  printf (" %lf%c", jmat[mu + N * nu], delim);
	}
    }

  int result = dsdgdt (workspace, s, jmat, jvec, hvec,
		       N, 1.0, dsdt);

  if (result == 0)
    {
      printf ("\n\nSample value of dsdt: \n [");
      for (mu = 0; mu < 3 * N; mu++)
	{
	  if (mu == 3 * N - 1)
	    delim = ']';
	  else
	    delim = ',';
	  printf (" %lf%c", dsdt[mu], delim);
	}
      printf ("\n\nSample value of dcdt: \n [ ");
      for (mu = 3 * N; mu < 9 * N * N + 3 * N; mu++)
	{
	  if (mu == 9 * N * N + 3 * N - 1)
	    delim = ']';
	  else
	    delim = ',';
	  printf (" %lf%c", dsdt[mu], delim);
	}
    }
  printf ("\n");
  free (workspace);
  free (s);
  free (dsdt);
  return 0;
}
