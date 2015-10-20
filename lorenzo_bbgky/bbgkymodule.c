#include <Python.h>
#include <numpy/arrayobject.h>
#include "lorenzo_bbgky.h"


static PyObject *
wrap_bbgky (PyObject * self, PyObject * args)
{
  PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
  PyObject *out = NULL;

  PyObject *s = NULL, *hopmat = NULL, *jvec = NULL, *hvec = NULL;
  PyObject *dsdt = NULL;
  double drv[1], latsize[1], norm[1];

  if (!PyArg_ParseTuple
      (args, "OOOOdddO!", &arg1, &arg2, &arg3, &arg4, drv, latsize, norm,
       &PyArray_Type, &out))
    return NULL;

  s = PyArray_FROM_OTF (arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (s == NULL)
    return NULL;
  hopmat = PyArray_FROM_OTF (arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (hopmat == NULL)
    goto fail;
  jvec = PyArray_FROM_OTF (arg3, NPY_DOUBLE, NPY_IN_ARRAY);
  if (jvec == NULL)
    goto fail;
  hvec = PyArray_FROM_OTF (arg4, NPY_DOUBLE, NPY_IN_ARRAY);
  if (hvec == NULL)
    goto fail;

  dsdt = PyArray_FROM_OTF (out, NPY_DOUBLE, NPY_INOUT_ARRAY);
  if (dsdt == NULL)
    goto fail;

  /* code that makes use of arguments */
  /* You will probably need at least
     nd = PyArray_NDIM(<..>)    -- number of dimensions
     dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
     showing length in each dim.
     dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

     If an error occurs goto fail.
   */

  Py_DECREF (s);
  Py_DECREF (dsdt);
  Py_DECREF (hopmat);
  Py_DECREF (jvec);
  Py_DECREF (hvec);
  Py_DECREF (dsdt);
  Py_INCREF (Py_None);
  return Py_None;

fail:
  Py_XDECREF (s);
  Py_XDECREF (dsdt);
  Py_XDECREF (hopmat);
  Py_XDECREF (jvec);
  Py_XDECREF (hvec);
  PyArray_XDECREF_ERR (dsdt);
  return NULL;
}

static PyMethodDef ModuleMethods[] = {
  {"bbgky", wrap_bbgky, METH_VARARGS | METH_KEYWORDS,
   "Executes the RHS of the bbgky dynamics"},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
initlorenzo_bbgky (void)
{
  (void) Py_InitModule ("lorenzo_bbgky", ModuleMethods);
  import_array ();
}
