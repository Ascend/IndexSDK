/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */


#ifndef ASCEND_TENSOR_INCLUDED
#define ASCEND_TENSOR_INCLUDED

#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/AscendMemory.h>
#include <cassert>
#include <initializer_list>

namespace ascendSearch {
const int DIMS_1 = 1;
const int DIMS_2 = 2;
const int DIMS_3 = 3;
const int DIMS_4 = 4;
const int DIMS_5 = 5;

namespace traits {
template<typename T>
struct RestrictPtrTraits {
    using PtrType = T *__restrict__;
};

template<typename T>
struct DefaultPtrTraits {
    using PtrType = T*;
};
} // namespace traits

template<typename T,
          int Dim,
          typename IndexT,
          template<typename U> class PtrTraits>
class AscendTensor;

namespace detail {
template<typename TensorType, int Subdim, template<typename U> class PtrTraits>
class SubTensor;
}

template<typename T,
          int Dim,
          typename IndexT = int,
          template<typename U> class PtrTraits = traits::DefaultPtrTraits>
class AscendTensor {
public:
    using DataType = T;
    using IndexType = IndexT;
    using DataPtrType = typename PtrTraits<T>::PtrType;
    using TensorType = AscendTensor<T, Dim, IndexT, PtrTraits>;

    enum NumDim {
        NUM_DIM = Dim
    };

    // Default constructor
    AscendTensor();

    // Destructor
    ~AscendTensor();

    // Copy constructor
    AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &t);

    // Move constructor
    AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &&t);

    // Copy assignment
    AscendTensor<T, Dim, IndexT, PtrTraits> &operator = (const AscendTensor<T, Dim, IndexT, PtrTraits> &t);

    // Move assignment
    AscendTensor<T, Dim, IndexT, PtrTraits> &operator = (AscendTensor<T, Dim, IndexT, PtrTraits> &&t);

    // Constructs a tensor of the given size and stride, referencing a
    // memory region we do not own
    AscendTensor(DataPtrType data, const IndexT sizes[Dim]);
    AscendTensor(DataPtrType data, std::initializer_list<IndexT> sizes);

    // Constructs a tensor of the given size and stride, referencing a
    // memory region we do not own
    AscendTensor(DataPtrType data, const IndexT sizes[Dim], const size_t strides[Dim]);

    // Constructs a tensor of the given size, allocating memory for it locally
    explicit AscendTensor(const IndexT sizes[Dim]);
    explicit AscendTensor(std::initializer_list<IndexT> sizes);

    // Constructs a tensor of the given size, reserving a temporary
    // memory reservation via a memory manager.
    // The memory reservation should be ordered with respect to the
    // given stream.
    AscendTensor(AscendMemory &m, const IndexT sizes[Dim], aclrtStream stream);
    AscendTensor(AscendMemory &m, std::initializer_list<IndexT> sizes, aclrtStream stream);

    // Copies a tensor into ourselves; sizes must match
    void copyFrom(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream,
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE) const;

    // Copies ourselves into a tensor; sizes must match
    void copyTo(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream,
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE) const;

    // Copies a tensor into ourselves; sizes must match
    void copyFromSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t,
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE) const;

    // Copies ourselves into a tensor; sizes must match
    void copyToSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t,
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE) const;

    // Call to zero out memory
    AscendTensor<T, Dim, IndexT, PtrTraits> &zero();

    // Cast to a tensor of a different type of the same size and
    // stride. U and our type T must be of the same size
    template<typename U>
    AscendTensor<U, Dim, IndexT, PtrTraits> cast();

    // Const cast to a tensor of a different type of the same size and
    // stride. U and our type T must be of the same size
    template<typename U>
    AscendTensor<U, Dim, IndexT, PtrTraits> cast() const;

    inline int dimNum() const;

    inline const IndexT *sizes() const;

    inline const size_t *strides() const;

    DataPtrType data() const;

    DataPtrType end() const;

    IndexT getSize(int i) const;

    size_t getStride(int i) const;

    size_t numElements() const;

    size_t getSizeInBytes() const;

    void resetDataPtr();

    void initValue(DataType val) const;

    // Returns a tensor that is a view of the `SubDim`-dimensional slice
    // of this tensor, starting at `at`.
    template<int SubDim>
    AscendTensor<T, SubDim, IndexT, PtrTraits> view(DataPtrType data) const;

    // Returns a tensor that is a view of the `SubDim`-dimensional slice
    // of this tensor, starting where our data begins
    template<int SubDim>
    AscendTensor<T, SubDim, IndexT, PtrTraits> view() const;

    // Returns a view of the given tensor expressed as a tensor of a
    // different number of dimensions.
    // Only works if we are contiguous.
    template<int NewDim>
    AscendTensor<T, NewDim, IndexT, PtrTraits> view(std::initializer_list<IndexT> sizes) const;

    // Returns a read/write view of a portion of our tensor.
    detail::SubTensor<TensorType, Dim - 1, PtrTraits> operator[](IndexT index);

    // Returns a const view of a portion of our tensor.
    const detail::SubTensor<TensorType, Dim - 1, PtrTraits> operator[](IndexT index) const;

private:
    DataPtrType dataPtr;

    size_t strideArray[Dim] = {0};

    IndexT sizeArray[Dim] = {0};

    enum class AllocState {
        ALLOC_STATE_OWNER = 0,
        ALLOC_STATE_NOT_OWNER = 1,
        ALLOC_STATE_RESERVATION = 2
    };

    AllocState state;
    AscendMemoryReservation reservation;
};

namespace detail {
template<typename TensorType, template<typename U> class PtrTraits>
class SubTensor<TensorType, 0, PtrTraits> {
public:
    SubTensor<TensorType, 0, PtrTraits> operator = (typename TensorType::DataType val);

    operator typename TensorType::DataType &();

    operator const typename TensorType::DataType &() const;

    typename TensorType::DataType *operator&();

    const typename TensorType::DataType *operator&() const;

    // Returns a raw accessor to our slice.
    typename TensorType::DataPtrType data();

    // Returns a raw accessor to our slice (const).
    const typename TensorType::DataPtrType data() const;

    typename TensorType::DataType value();

    const typename TensorType::DataType value() const;

    void value(typename TensorType::DataType val);

    // Cast to a different datatype.
    template<typename T>
    T &as() const ;

    // Cast to a different datatype(const).
    template<typename T>
    const T &as() const;

    // Cast to a different datatype
    template<typename T>
    typename PtrTraits<T>::PtrType dataAs();

    // Cast to a different datatype(const)
    template<typename T>
    typename PtrTraits<const T>::PtrType dataAs() const;

    // defautl destructor function
    ~SubTensor() {}

protected:
    friend class SubTensor<TensorType, 1, PtrTraits>;

    friend class AscendTensor<typename TensorType::DataType,
                              1,
                              typename TensorType::IndexType,
                              PtrTraits>;

    SubTensor(TensorType &t, typename TensorType::DataPtrType d);

    TensorType &tensor;

    typename TensorType::DataPtrType const dataPtr;
};

// A `SubDim`-rank slice of a parent Tensor
template<typename TensorType, int SubDim, template<typename U> class PtrTraits>
class SubTensor {
public:
    // Returns a view of the data located at our offset (the dimension
    // `SubDim` - 1 tensor).
    inline SubTensor<TensorType, SubDim - 1, PtrTraits> operator[](typename TensorType::IndexType index);

    // Returns a view of the data located at our offset (the dimension
    // `SubDim` - 1 tensor)(const).
    inline const SubTensor<TensorType, SubDim - 1, PtrTraits> operator[](typename TensorType::IndexType index) const;

    // operator& returning T*
    typename TensorType::DataType *operator&();

    // const operator& returning const T*
    const typename TensorType::DataType *operator&() const;

    // Returns a raw accessor to our slice.
    typename TensorType::DataPtrType data();

    // Returns a raw accessor to our slice (const).
    const typename TensorType::DataPtrType data() const;

    // Cast to a different datatype.
    template<typename T>
    T &as() const;

    // Cast to a different datatype(const).
    template<typename T>
    const T &as() const;

    // Cast to a different datatype
    template<typename T>
    typename PtrTraits<T>::PtrType dataAs();

    // Cast to a different datatype(const)
    template<typename T>
    typename PtrTraits<const T>::PtrType dataAs() const;

    AscendTensor<typename TensorType::DataType, SubDim, typename TensorType::IndexType, PtrTraits>
    view();

    // defautl destructor function
    ~SubTensor() {}

protected:
    // One dimension greater can create us
    friend class SubTensor<TensorType, SubDim + 1, PtrTraits>;

    /// Our parent tensor can create us
    friend class AscendTensor<typename TensorType::DataType,
                              TensorType::NUM_DIM,
                              typename TensorType::IndexType,
                              PtrTraits>;

    SubTensor(TensorType &t, typename TensorType::DataPtrType data);

    // The tensor we're referencing
    TensorType &tensor;

    // Where our value is located
    typename TensorType::DataPtrType const dataPtr;
};
} // namespace detail
} // namespace ascendSearch
#include <ascenddaemon/utils/AscendTensorInl.h>

#endif // ASCEND_TENSOR_INCLUDED
