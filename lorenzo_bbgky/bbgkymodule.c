#include <Python.h>
#include <numpy/arrayobject.h>
#include "lorenzo_bbgky.h"


static PyObject *
wrap_bbgky (PyObject * self, PyObject * args)
{
  PyObject *arg0 = NULL;
  PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
  PyObject *out = NULL;

  PyObject *workspace = NULL;
  PyObject *s = NULL, *hopmat = NULL, *jvec = NULL, *hvec = NULL;
  PyObject *dsdt = NULL;
  double drv, norm;
  int latsize, ret;

  if (!PyArg_ParseTuple
      (args, "OOOOOdidO!", &arg0, &arg1, &arg2, &arg3, &arg4, &drv, &latsize,
       &norm, &PyArray_Type, &out))
    return NULL;

  workspace = PyArray_FROM_OTF (arg0, NPY_DOUBLE, NPY_IN_ARRAY);
  if (workspace == NULL)
    return NULL;
  s = PyArray_FROM_OTF (arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (s == NULL)
    return NULL;
  if (PyArray_NDIM (s) != 1)
    goto fail;
  hopmat = PyArray_FROM_OTF (arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if ((hopmat == NULL) || (PyArray_NDIM (hopmat) != 1))
    goto fail;
  jvec = PyArray_FROM_OTF (arg3, NPY_DOUBLE, NPY_IN_ARRAY);
  if ((jvec == NULL) || (PyArray_NDIM (jvec) != 1))
    goto fail;
  hvec = PyArray_FROM_OTF (arg4, NPY_DOUBLE, NPY_IN_ARRAY);
  if ((hvec == NULL) || (PyArray_NDIM (hvec) != 1))
    goto fail;

  dsdt = PyArray_FROM_OTF (out, NPY_DOUBLE, NPY_INOUT_ARRAY);
  if ((dsdt == NULL) || (PyArray_NDIM (dsdt) != 1))
    goto fail;

  /* code that makes use of arguments */
  /* You will probably need at least
     nd = PyArray_NDIM(<..>)    -- number of dimensions
     d  ims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
     showing length in each dim. */

  double *wspace_ptr;
  double *s_ptr, *hopmat_ptr, *jvec_ptr, *hvec_ptr, *dsdt_ptr;

  wspace_ptr = (double *) PyArray_DATA (workspace);

  s_ptr = (double *) PyArray_DATA (s);
  hopmat_ptr = (double *) PyArray_DATA (hopmat);
  jvec_ptr = (double *) PyArray_DATA (jvec);
  hvec_ptr = (double *) PyArray_DATA (hvec);
  dsdt_ptr = (double *) PyArray_DATA (dsdt);

  ret =
    dsdgdt ((double *) wspace_ptr, (double *) s_ptr, (double *) hopmat_ptr,
	    (double *) jvec_ptr, (double *) hvec_ptr, drv, latsize, norm,
	    (double *) dsdt_ptr);
  if (ret != 0)
    goto fail;

  Py_DECREF (workspace);
  Py_DECREF (s);
  Py_DECREF (hopmat);
  Py_DECREF (jvec);
  Py_DECREF (hvec);
  Py_DECREF (dsdt);
  Py_INCREF (Py_None);
  return Py_None;

fail:
  Py_XDECREF (workspace);
  Py_XDECREF (s);
  Py_XDECREF (hopmat);
  Py_XDECREF (jvec);
  Py_XDECREF (hvec);
  Py_XDECREF (dsdt);
  return NULL;
}

static PyMethodDef ModuleMethods[] = {
  {"bbgky", wrap_bbgky, METH_VARARGS | METH_KEYWORDS,
   "bbgky(s, jmat, jvec, hvec, drive, N, norm, dsdt)\n\\n\
C code with cblas dependency that optimally computes the RHS of the bbgky dynamics. \n Call this function from python as lorenzo_bbgky.bbgky(args)\n Arguments in the following order. All are either ints, doubles or 1d numpy arrays:n w\t-\t Workspace that is a Numpy array of minimum size 3*N+9*N*N\n s\t-\tNumpy array of all spins sx, sy, sz (vecs of size N) and correlations matrices (size N X N)\n \t\tflattened as [sx, sy, sz, gxx, gxy, gxz, gyx, gyy, gyz, gzx, gzy, gzz]),\n jmat\t-\tHopping matrix (NXN), flattened to 1d array,\n jvec\t-\tVector of hopping amplitudes [jx,jy,jz],\n hvec\t-\tVector of fields [hx,hy,hz],\n drive\t-\tPeriodic drive at the time when called. Set to unity to disable,\n N\t-\tLattice size,\n norm\t-\tNormalization of jvec,\n dsdt\t-\tOutput numpy array (same structure as s) "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
initlorenzo_bbgky (void)
{
  (void) Py_InitModule ("lorenzo_bbgky", ModuleMethods);
  import_array ();
}
