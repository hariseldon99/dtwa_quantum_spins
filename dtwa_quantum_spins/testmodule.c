#include <Python.h>

#include "libtest.h"

static PyObject* wrap_test_get_data_nulls(PyObject* self)
{
    char *s;
    int l;
 
    s = test_get_data_nulls(&l);
    return PyString_FromStringAndSize(s, l);
}
 
static PyObject* wrap_test_get_data(PyObject* self, PyObject* args)
{
    char *s;
    unsigned int l;
 
    if (!PyArg_ParseTuple(args, "I", &l))
        return NULL;
 
    s = test_get_data(l);
    return PyString_FromStringAndSize(s, l);
}
 
static PyMethodDef ModuleMethods[] =
{
     {"test_get_data", wrap_test_get_data, METH_VARARGS, "Get a string of variable length"},
     {"test_get_data_nulls", wrap_test_get_data_nulls, METH_NOARGS, "Get a string of fixed length with embedded nulls"},
     {NULL, NULL, 0, NULL},
};
 
PyMODINIT_FUNC
 
inittestmodule(void)
{
     (void) Py_InitModule("testmodule", ModuleMethods);
}
