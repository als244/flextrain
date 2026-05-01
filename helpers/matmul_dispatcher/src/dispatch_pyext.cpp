// CPython extension exposing a fast-path entry point to the cuBLASLt
// dispatcher. The existing ctypes-based ``matmul`` method has ~3.9us
// of Python preamble + ~1.3us of ctypes FFI per call; this module's
// ``matmul_fast`` skips both by:
//   * Using METH_FASTCALL (no tuple-packing of args).
//   * Taking pre-extracted raw values (data ptrs, leading dims, dims,
//     transpose flags, dtype enum, alpha/beta) as Python ints/floats —
//     the caller does tensor introspection ONCE outside the hot loop.
//   * Calling the existing C ABI ``dispatch_matmul`` directly.
//
// Math contract is identical to ``CublasLtDispatcher.matmul`` — same
// algo cache, same workspace, same kernels.
//
// Used by the MoE LoRA per-expert callback (see
// flextrain/nn/layers/lora_wrapper.py:_make_moe_callback) where the
// callback fires hundreds of times per layer and Python-side overhead
// dominates.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#include "dispatch.h"

// matmul_fast(ctx_capsule_or_int, stream_ptr,
//             M, N, K,
//             A_ptr, lda, trans_a,
//             B_ptr, ldb, trans_b,
//             C_ptr, ldc,
//             D_ptr, ldd,
//             ws_ptr, ws_bytes,
//             a_dt, b_dt, c_dt, d_dt, compute_dt,
//             alpha, beta) -> None
//
// All pointer/int args are Python ints (not ctypes wrappers). On error
// raises RuntimeError. Returns None on success.
static PyObject* matmul_fast(
    PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs
) {
    if (nargs != 24) {
        PyErr_Format(
            PyExc_TypeError,
            "matmul_fast expected 24 positional args, got %zd", nargs
        );
        return nullptr;
    }

    // ctx is passed as a plain int (the ctypes c_void_p value the
    // CublasLtDispatcher created the context with).
    DispatchContext* ctx =
        reinterpret_cast<DispatchContext*>(PyLong_AsVoidPtr(args[0]));
    if (!ctx) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "matmul_fast: ctx is NULL");
        }
        return nullptr;
    }

    intptr_t stream_ptr = (intptr_t)PyLong_AsVoidPtr(args[1]);

    int M = (int)PyLong_AsLong(args[2]);
    int N = (int)PyLong_AsLong(args[3]);
    int K = (int)PyLong_AsLong(args[4]);

    intptr_t a_ptr = (intptr_t)PyLong_AsVoidPtr(args[5]);
    int lda        = (int)PyLong_AsLong(args[6]);
    int trans_a    = (int)PyLong_AsLong(args[7]);

    intptr_t b_ptr = (intptr_t)PyLong_AsVoidPtr(args[8]);
    int ldb        = (int)PyLong_AsLong(args[9]);
    int trans_b    = (int)PyLong_AsLong(args[10]);

    intptr_t c_ptr = (intptr_t)PyLong_AsVoidPtr(args[11]);
    int ldc        = (int)PyLong_AsLong(args[12]);

    intptr_t d_ptr = (intptr_t)PyLong_AsVoidPtr(args[13]);
    int ldd        = (int)PyLong_AsLong(args[14]);

    intptr_t ws_ptr = (intptr_t)PyLong_AsVoidPtr(args[15]);
    size_t ws_bytes = (size_t)PyLong_AsSize_t(args[16]);

    int a_dt       = (int)PyLong_AsLong(args[17]);
    int b_dt       = (int)PyLong_AsLong(args[18]);
    int c_dt       = (int)PyLong_AsLong(args[19]);
    int d_dt       = (int)PyLong_AsLong(args[20]);
    int compute_dt = (int)PyLong_AsLong(args[21]);

    float alpha = (float)PyFloat_AsDouble(args[22]);
    float beta  = (float)PyFloat_AsDouble(args[23]);

    if (PyErr_Occurred()) {
        return nullptr;  // arg parsing already raised
    }

    int rc = dispatch_matmul(
        ctx, stream_ptr,
        M, N, K,
        a_ptr, lda,
        b_ptr, ldb,
        c_ptr, ldc,
        d_ptr, ldd,
        ws_ptr, ws_bytes,
        a_dt, b_dt, c_dt, d_dt, compute_dt,
        alpha, beta,
        trans_a, trans_b
    );
    if (rc != 0) {
        PyErr_Format(
            PyExc_RuntimeError,
            "matmul_fast: dispatch_matmul failed (rc=%d)", rc
        );
        return nullptr;
    }
    Py_RETURN_NONE;
}


static PyMethodDef methods[] = {
    {
        "matmul_fast",
        (PyCFunction)matmul_fast,
        METH_FASTCALL,
        "Fast cuBLASLt dispatch with pre-extracted args (no ctypes, no "
        "tensor introspection on hot path). See module docstring."
    },
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_dispatch_pyext",  // m_name
    "Fast-path cuBLASLt dispatcher entry (CPython extension).",  // m_doc
    -1,                 // m_size
    methods,
    nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit__dispatch_pyext(void) {
    return PyModule_Create(&moduledef);
}
