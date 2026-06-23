/*
 * Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

%module ascendfaiss;


%include <stdint.i>

typedef uint64_t size_t;


#define __restrict


/*******************************************************************
 * Copied verbatim to wrapper. Contains the C++-visible includes, and
 * the language includes for their respective matrix libraries.
 *******************************************************************/

%{


#include <stdint.h>

#ifdef SWIGPYTHON

#undef popcount64

#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#endif

#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendIndexIVFSP.h>
#include <faiss/ascend/custom/AscendIndexIVFSQT.h>
#include <faiss/ascend/AscendIndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/ascend/utils/Version.h>


%}

/********************************************************
 * GIL manipulation and exception handling
 ********************************************************/

#ifdef SWIGPYTHON
// %catches(faiss::FaissException);


// Python-specific: release GIL by default for all functions
%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(faiss::FaissException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc & ba) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    } catch(const std::exception& ex) {
        PyEval_RestoreThread(_save);
        std::string what = std::string("C++ exception ") + ex.what();
        PyErr_SetString(PyExc_RuntimeError, what.c_str());
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}

#endif


/*******************************************************************
 * Types of vectors we want to manipulate at the scripting language
 * level.
 *******************************************************************/

// simplified interface for vector
namespace std {

    template<class T>
    class vector {
    public:
        vector();
        void push_back(T);
        void clear();
        T * data();
        size_t size();
        T at (size_t n) const;
        T & operator [] (size_t n);
        void resize (size_t n);
        void swap (vector<T> & other);
    };
};

%include <std_string.i>
%include <std_pair.i>


%include <std_map.i>
%include <std_shared_ptr.i>

// primitive array types
%template(FloatVector) std::vector<float>;
%template(DoubleVector) std::vector<double>;
%template(ByteVector) std::vector<uint8_t>;
%template(CharVector) std::vector<char>;
%template(UInt64Vector) std::vector<unsigned long>;
%template(LongVector) std::vector<long>;
%template(IntVector) std::vector<int>;
%template(FaissIdxVector) std::vector<faiss::idx_t>;

%template(FloatVectorVector) std::vector<std::vector<float> >;
%template(ByteVectorVector) std::vector<std::vector<unsigned char> >;
%template(CharVectorVector) std::vector<std::vector<char> >;
%template(LongVectorVector) std::vector<std::vector<long> >;


%ignore faiss::ascend::AscendIndexConfig::AscendIndexConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFConfig::AscendIndexIVFConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFSQConfig::AscendIndexIVFSQConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexInt8Config::AscendIndexInt8Config(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexInt8FlatConfig::AscendIndexInt8FlatConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexSQConfig::AscendIndexSQConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexFlatConfig::AscendIndexFlatConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFSPConfig::AscendIndexIVFSPConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexIVFSQTConfig::AscendIndexIVFSQTConfig(std::initializer_list<int>, int);
%ignore faiss::ascend::AscendIndexBinaryFlatConfig::AscendIndexBinaryFlatConfig(std::initializer_list<int>, int);


%include <faiss/MetricType.h>
%include <faiss/impl/platform_macros.h>
%include <faiss/Index.h>
%include <faiss/IndexBinary.h>
%include <faiss/Clustering.h>
%include <faiss/ascend/AscendIndex.h>
%include <faiss/ascend/AscendIndexIVF.h>
%include <faiss/ascend/AscendIndexIVFSQ.h>
%include <faiss/ascend/AscendIndexInt8.h>
%include <faiss/ascend/AscendIndexInt8Flat.h>
%include <faiss/ascend/AscendIndexFlat.h>
%include <faiss/ascend/AscendIndexSQ.h>
%include <faiss/ascend/AscendIndexIVFSP.h>
%include <faiss/ascend/custom/AscendIndexIVFSQT.h>
%include <faiss/ascend/AscendIndexBinaryFlat.h>
%include <faiss/clone_index.h>
%include <faiss/ascend/AscendClonerOptions.h>
%include <faiss/ascend/utils/Version.h>


%ignore faiss::InvertedListScanner;
%ignore faiss::ScalarQuantizer::SQDistanceComputer;
%ignore faiss::ScalarQuantizer::Quantizer;
%ignore faiss::ScalarQuantizer::select_InvertedListScanner;
%ignore faiss::ScalarQuantizer::get_distance_computer;

%include  <faiss/impl/ScalarQuantizer.h>

%ignore faiss::BufferList::Buffer;
%ignore faiss::RangeSearchPartialResult::QueryResult;
%ignore faiss::IDSelectorBatch::set;
%ignore faiss::IDSelectorBatch::bloom;
%ignore faiss::InterruptCallback::instance;
%ignore faiss::InterruptCallback::lock;

%include <faiss/impl/AuxIndexStructures.h>
%include <faiss/impl/IDSelector.h>


#ifdef SWIGPYTHON

%define DOWNCAST(subclass)
    if (dynamic_cast<faiss::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## subclass,$owner);
    } else
%enddef

%define DOWNCAST2(subclass, longname)
    if (dynamic_cast<faiss::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## longname,$owner);
    } else
%enddef

%define DOWNCAST_ASCEND(subclass)
    if (dynamic_cast<faiss::ascend::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ascend__ ## subclass,$owner);
    } else
%enddef

#endif



// Subclasses should appear before their parent
%typemap(out) faiss::Index * {

    DOWNCAST_ASCEND ( AscendIndexFlat )
    DOWNCAST_ASCEND ( AscendIndexIVFSQ )
    DOWNCAST_ASCEND ( AscendIndexSQ )
    DOWNCAST_ASCEND ( AscendIndexIVFSP )
    DOWNCAST_ASCEND ( AscendIndexIVFSQT )

    // default for non-recognized classes
    DOWNCAST ( Index )
    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}

%typemap(out) faiss::IndexBinary * {

    DOWNCAST_ASCEND ( AscendIndexBinaryFlat )

    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}

%typemap(out) faiss::ascend::AscendIndexInt8 * {

    DOWNCAST_ASCEND ( AscendIndexInt8Flat )

    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}


// just to downcast pointers that come from elsewhere (eg. direct
// access to object fields)
%inline %{
faiss::Index * downcast_index (faiss::Index *index)
{
    return index;
}
%}


%newobject index_ascend_to_cpu;
%newobject index_cpu_to_ascend;
%newobject index_int8_ascend_to_cpu;
%newobject index_int8_cpu_to_ascend;
%include <faiss/ascend/AscendCloner.h>


/*******************************************************************
 * Python specific: numpy array <-> C++ pointer interface
 *******************************************************************/

#ifdef SWIGPYTHON

%{
PyObject *swig_ptr (PyObject *a)
{

    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}

%}


%init %{
    /* needed, else crash at runtime */
    import_array();

%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)

%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef

REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(unsigned char, NPY_UINT8);
REV_SWIG_PTR(char, NPY_INT8);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);

#endif

// End of file...
