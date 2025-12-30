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

#include <assert.h>
#include <initializer_list>

#include "npu/common/utils/MemorySpace.h"
#include "npu/common/utils/AscendUtils.h"
#include "npu/common/utils/AscendMemory.h"

namespace ascendSearchacc {
const int DIMS_1 = 1;
const int DIMS_2 = 2;
const int DIMS_3 = 3;
const int DIMS_4 = 4;
const int DIMS_5 = 5;

namespace traits {
template <typename T>
struct RestrictPtrTraits {
    typedef T *__restrict__ PtrType;
};

template <typename T>
struct DefaultPtrTraits {
    typedef T *PtrType;
};
}  // namespace traits

template <typename T, int Dim, typename IndexT, template <typename U> class PtrTraits>
class AscendTensor;

namespace detail {
template <typename TensorType, int Subdim, template <typename U> class PtrTraits>
class SubTensor;
}

class AscendTensorBase {
public:
    virtual void *getVoidData() const = 0;
    virtual size_t getSizeInBytes() const = 0;
    virtual ~AscendTensorBase() = default;
};

template <typename T, int Dim, typename IndexT = int, template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class AscendTensor : public AscendTensorBase {
public:
    typedef T DataType;
    typedef IndexT IndexType;
    typedef typename PtrTraits<T>::PtrType DataPtrType;
    typedef AscendTensor<T, Dim, IndexT, PtrTraits> TensorType;

    enum {
        NUM_DIM = Dim
    };

    // Default constructor
    AscendTensor() : dataPtr(nullptr), state(ALLOC_STATE_NOT_OWNER)
    {
        static_assert(Dim > 0, "must have > 0 dimensions");
        for (int i = 0; i < Dim; ++i) {
            sizeArray[i] = 0;
            strideArray[i] = 1;
        }
    }

    // Destructor
    ~AscendTensor()
    {
        if (state == ALLOC_STATE_OWNER && this->dataPtr != nullptr) {
            FreeMemorySpace(MemorySpace::DEVICE, (void *)this->dataPtr);
            this->dataPtr = nullptr;
        }
    }

    // Copy constructor
    AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &t) { this->operator=(t); }

    // Move constructor
    AscendTensor(AscendTensor<T, Dim, IndexT, PtrTraits> &&t) { this->operator=(std::move(t)); }

    // Copy assignment
    AscendTensor<T, Dim, IndexT, PtrTraits> &operator=(const AscendTensor<T, Dim, IndexT, PtrTraits> &t)
    {
        this->dataPtr = nullptr;
        for (int i = 0; i < Dim; ++i) {
            sizeArray[i] = t.sizeArray[i];
            strideArray[i] = t.strideArray[i];
        }

        this->state = ALLOC_STATE_OWNER;
        allocMemorySpace(MemorySpace::DEVICE, &this->dataPtr, t.getSizeInBytes());
        auto ret = memcpy_s((void *)this->data(), this->getSizeInBytes(), (void *)t.data(), t.getSizeInBytes());
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Mem operator error %d", ret);

        return *this;
    }

    // Move assignment
    AscendTensor<T, Dim, IndexT, PtrTraits> &operator=(AscendTensor<T, Dim, IndexT, PtrTraits> &&t)
    {
        if (this->state == ALLOC_STATE_OWNER && this->dataPtr != nullptr) {
            FreeMemorySpace(MemorySpace::DEVICE, (void *)this->dataPtr);
        }

        dataPtr = t.dataPtr;
        for (int i = 0; i < Dim; ++i) {
            sizeArray[i] = t.sizeArray[i];
            strideArray[i] = t.strideArray[i];
            t.sizeArray[i] = 0;
            t.strideArray[i] = 0;
        }

        t.dataPtr = nullptr;

        this->state = t.state;
        t.state = ALLOC_STATE_NOT_OWNER;
        this->reservation = std::move(t.reservation);

        return *this;
    }

    // Constructs a tensor of the given size and stride, referencing a
    // memory region we do not own
    AscendTensor(DataPtrType data, const IndexT sizes[Dim])
        : dataPtr(data), state(ALLOC_STATE_NOT_OWNER)
    {
        static_assert(Dim > 0, "must have > 0 dimensions");

        for (int i = 0; i < Dim; ++i) {
            sizeArray[i] = sizes[i];
        }

        strideArray[Dim - 1] = 1;
        const int offset = 2;
        for (int i = Dim - offset; i >= 0; --i) {
            strideArray[i] = strideArray[i + 1] * static_cast<size_t>(sizeArray[i + 1]);
        }
    }

    AscendTensor(DataPtrType data, std::initializer_list<IndexT> sizes)
        : dataPtr(data), state(ALLOC_STATE_NOT_OWNER)
    {
        static_assert(Dim > 0, "must have > 0 dimensions");

        int i = 0;
        for (auto s : sizes) {
            sizeArray[i] = s;
            i++;
        }

        strideArray[Dim - 1] = 1;
        const int offset = 2;
        for (i = Dim - offset; i >= 0; --i) {
            strideArray[i] = strideArray[i + 1] * static_cast<size_t>(sizeArray[i + 1]);
        }
    }

    // Constructs a tensor of the given size and stride, referencing a
    // memory region we do not own
    AscendTensor(DataPtrType data, const IndexT sizes[Dim], const size_t strides[Dim])
        : dataPtr(data), state(ALLOC_STATE_NOT_OWNER)
    {
        for (int i = 0; i < Dim; ++i) {
            sizeArray[i] = sizes[i];
            strideArray[i] = strides[i];
        }
    }

    // Constructs a tensor of the given size, allocating memory for it locally
    AscendTensor(const IndexT sizes[Dim]) : AscendTensor(nullptr, sizes)
    {
        this->state = ALLOC_STATE_OWNER;
        allocMemorySpace(MemorySpace::DEVICE, &this->dataPtr, this->getSizeInBytes());
    }

    AscendTensor(std::initializer_list<IndexT> sizes) : AscendTensor(nullptr, sizes)
    {
        this->state = ALLOC_STATE_OWNER;
        allocMemorySpace(MemorySpace::DEVICE, &this->dataPtr, this->getSizeInBytes());
    }

    // Constructs a tensor of the given size, reserving a temporary
    // memory reservation via a memory manager.
    // The memory reservation should be ordered with respect to the
    // given stream.
    AscendTensor(AscendMemory &m, const IndexT sizes[Dim], aclrtStream stream)
        : AscendTensor(nullptr, sizes)
    {
        this->state = ALLOC_STATE_RESERVATION;

        auto memory = m.getMemory(stream, this->getSizeInBytes());

        this->dataPtr = (T *)memory.get();
        reservation = std::move(memory);
    }

    AscendTensor(AscendMemory &m, std::initializer_list<IndexT> sizes, aclrtStream stream)
        : AscendTensor(nullptr, sizes)
    {
        this->state = ALLOC_STATE_RESERVATION;

        auto memory = m.getMemory(stream, this->getSizeInBytes());

        this->dataPtr = (T *)memory.get();
        reservation = std::move(memory);
    }

    // Copies a tensor into ourselves; sizes must match
    void copyFrom(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream,
                  aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE)
    {
        // Size must match
        ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

        if (t.numElements() > 0) {
            ASCEND_THROW_IF_NOT(this->data());
            ASCEND_THROW_IF_NOT(t.data());

            ACL_REQUIRE_OK(aclrtMemcpyAsync((void *)this->data(), this->getSizeInBytes(), (void *)t.data(),
                                            t.getSizeInBytes(), kind, stream));
        }
    }

    // Copies ourselves into a tensor; sizes must match
    void copyTo(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtStream stream,
                aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE)
    {
        // Size must match
        ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

        if (this->numElements() > 0) {
            ASCEND_THROW_IF_NOT(data());
            ASCEND_THROW_IF_NOT(t.data());

            ACL_REQUIRE_OK(aclrtMemcpyAsync((void *)t.data(), t.getSizeInBytes(), (void *)this->data(),
                                            this->getSizeInBytes(), kind, stream));
        }
    }

    // Copies a tensor into ourselves; sizes must match
    void copyFromSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE)
    {
        // Size must match
        ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

        if (t.numElements() > 0) {
            ASCEND_THROW_IF_NOT(this->data());
            ASCEND_THROW_IF_NOT(t.data());

            auto ret = aclrtMemcpy((void *)this->data(), this->getSizeInBytes(), (void *)t.data(), t.getSizeInBytes(),
                                   kind);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);
        }
    }

    // Copies ourselves into a tensor; sizes must match
    void copyToSync(AscendTensor<T, Dim, IndexT, PtrTraits> &t, aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE)
    {
        // Size must match
        ASCEND_THROW_IF_NOT(this->getSizeInBytes() == t.getSizeInBytes());

        if (this->numElements() > 0) {
            ASCEND_THROW_IF_NOT(data());
            ASCEND_THROW_IF_NOT(t.data());

            auto ret = aclrtMemcpy((void *)t.data(), t.getSizeInBytes(), (void *)this->data(), this->getSizeInBytes(),
                                   kind);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);
        }
    }

    // Call to zero out memory
    AscendTensor<T, Dim, IndexT, PtrTraits> &zero()
    {
        if (this->dataPtr != nullptr && this->getSizeInBytes() > 0) {
#ifdef HOSTCPU
            auto error = aclrtMemset(this->dataPtr, this->getSizeInBytes(), 0, this->getSizeInBytes());
            ASCEND_THROW_IF_NOT_FMT(error == ACL_SUCCESS, "failed to memset (error %d)", (int)error);
#else
            auto error = memset_s(this->dataPtr, this->getSizeInBytes(), 0, this->getSizeInBytes());
            ASCEND_THROW_IF_NOT_FMT(error == EOK, "failed to memset (error %d)", (int)error);
#endif
        }

        return *this;
    }

    // Cast to a tensor of a different type of the same size and
    // stride. U and our type T must be of the same size
    template <typename U>
    AscendTensor<U, Dim, IndexT, PtrTraits> cast()
    {
        static_assert(sizeof(U) == sizeof(T), "cast must be to same size object");

        return AscendTensor<U, Dim, IndexT, PtrTraits>(reinterpret_cast<U *>(dataPtr), sizeArray, strideArray);
    }

    // Const cast to a tensor of a different type of the same size and
    // stride. U and our type T must be of the same size
    template <typename U>
    AscendTensor<U, Dim, IndexT, PtrTraits> cast() const
    {
        static_assert(sizeof(U) == sizeof(T), "cast must be to same size object");

        return AscendTensor<U, Dim, IndexT, PtrTraits>(reinterpret_cast<U *>(dataPtr), sizeArray, strideArray);
    }

    inline int dimNum() const { return NUM_DIM; }

    inline const IndexT *sizes() const { return sizeArray; }

    inline const size_t *strides() const { return strideArray; }

    DataPtrType data() const { return dataPtr; }

    void *getVoidData() const { return static_cast<void *>(dataPtr); }

    DataPtrType end() const { return data() + numElements(); }

    IndexT getSize(int i) const
    {
        ASCEND_THROW_IF_NOT(i >= 0);
        ASCEND_THROW_IF_NOT(i < Dim);
        return sizeArray[i];
    }

    size_t getStride(int i) const
    {
        ASCEND_THROW_IF_NOT(i >= 0);
        ASCEND_THROW_IF_NOT(i < Dim);
        return strideArray[i];
    }

    size_t numElements() const
    {
        size_t size = (size_t)getSize(0);

        for (int i = 1; i < Dim; ++i) {
            size *= (size_t)getSize(i);
        }

        return size;
    }

    size_t getSizeInBytes() const { return sizeof(T) * numElements(); }

    void resetDataPtr()
    {
        if (dataPtr != nullptr) {
            FreeMemorySpace(MemorySpace::DEVICE, (void *)dataPtr);
            dataPtr = nullptr;
        }
    }

    void initValue(DataType val)
    {
        if (this->numElements() > 0) {
            ASCEND_THROW_IF_NOT(this->data());

            std::fill_n(this->dataPtr, this->numElements(), val);
        }
    }

    // Returns a tensor that is a view of the `SubDim`-dimensional slice
    // of this tensor, starting at `at`.
    template <int SubDim>
    AscendTensor<T, SubDim, IndexT, PtrTraits> view(DataPtrType data)
    {
        ASCEND_THROW_IF_NOT_MSG(SubDim >= 1 && SubDim < Dim, "can only create view of lesser dim");

        IndexT viewSizes[SubDim];
        size_t viewStrides[SubDim];

        for (int i = 0; i < SubDim; ++i) {
            viewSizes[i] = sizeArray[Dim - SubDim + i];
            viewStrides[i] = strideArray[Dim - SubDim + i];
        }

        return AscendTensor<T, SubDim, IndexT, PtrTraits>(data, viewSizes, viewStrides);
    }

    // Returns a tensor that is a view of the `SubDim`-dimensional slice
    // of this tensor, starting where our data begins
    template <int SubDim>
    AscendTensor<T, SubDim, IndexT, PtrTraits> view() { return view<SubDim>(data()); }

    // Returns a view of the given tensor expressed as a tensor of a
    // different number of dimensions.
    // Only works if we are contiguous.
    template <int NewDim>
    AscendTensor<T, NewDim, IndexT, PtrTraits> view(std::initializer_list<IndexT> sizes)
    {
        ASCEND_THROW_IF_NOT(sizes.size() == NewDim);

        // Verify the sizes
        size_t curSize = numElements();
        size_t newSize = 1;

        for (auto sz : sizes) {
            newSize *= sz;
        }

        ASCEND_THROW_IF_NOT(curSize == newSize);
        return AscendTensor<T, NewDim, IndexT, PtrTraits>(data(), sizes);
    }

    // Returns a read/write view of a portion of our tensor.
    detail::SubTensor<TensorType, Dim - 1, PtrTraits> operator[](IndexT index)
    {
        return detail::SubTensor<TensorType, Dim - 1, PtrTraits>(
            detail::SubTensor<TensorType, Dim, PtrTraits>(*this, data())[index]);
    }

    // Returns a const view of a portion of our tensor.
    const detail::SubTensor<TensorType, Dim - 1, PtrTraits> operator[](IndexT index) const
    {
        return detail::SubTensor<TensorType, Dim - 1, PtrTraits>(
            detail::SubTensor<TensorType, Dim, PtrTraits>(const_cast<AscendTensor::TensorType &>(*this),
                                                          data())[index]);
    }

private:
    DataPtrType dataPtr;

    size_t strideArray[Dim] = {};

    IndexT sizeArray[Dim] = {};

    enum AllocState {
        ALLOC_STATE_OWNER,
        ALLOC_STATE_NOT_OWNER,
        ALLOC_STATE_RESERVATION
    };

    AllocState state;
    AscendMemoryReservation reservation;
};

namespace detail {
template <typename TensorType, template <typename U> class PtrTraits>
class SubTensor<TensorType, 0, PtrTraits> {
public:
    SubTensor<TensorType, 0, PtrTraits> operator=(typename TensorType::DataType val)
    {
#ifdef HOSTCPU
        // ATTENTION: set value remote, should not be called frequently
        auto ret = aclrtMemcpy(dataPtr, sizeof(val), &val, sizeof(val), ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);
#else
        *dataPtr = val;
#endif
        return *this;
    }

    operator typename TensorType::DataType &() { return *dataPtr; }

    operator const typename TensorType::DataType &() const { return *dataPtr; }

    typename TensorType::DataType *operator&() { return dataPtr; }

    const typename TensorType::DataType *operator&() const { return dataPtr; }

    // Returns a raw accessor to our slice.
    typename TensorType::DataPtrType data() { return dataPtr; }

    // Returns a raw accessor to our slice (const).
    const typename TensorType::DataPtrType data() const { return dataPtr; }

    typename TensorType::DataType value() { return *dataPtr; }

    const typename TensorType::DataType value() const { return *dataPtr; }

    void value(typename TensorType::DataType val) { *dataPtr = val; }

    // Cast to a different datatype.
    template <typename T>
    T &as() { return *dataAs<T>(); }

    // Cast to a different datatype(const).
    template <typename T>
    const T &as() const { return *dataAs<T>(); }

    // Cast to a different datatype
    template <typename T>
    typename PtrTraits<T>::PtrType dataAs() { return reinterpret_cast<typename PtrTraits<T>::PtrType>(dataPtr); }

    // Cast to a different datatype(const)
    template <typename T>
    typename PtrTraits<const T>::PtrType dataAs() const { return reinterpret_cast<typename PtrTraits<const T>::PtrType>(dataPtr); }

    // defautl destructor function
    ~SubTensor() {}

protected:
    friend class SubTensor<TensorType, 1, PtrTraits>;

    friend class AscendTensor<typename TensorType::DataType, 1, typename TensorType::IndexType, PtrTraits>;

    SubTensor(TensorType &t, typename TensorType::DataPtrType d) : tensor(t), dataPtr(d) {}

    TensorType &tensor;

    typename TensorType::DataPtrType const dataPtr;
};

// A `SubDim`-rank slice of a parent Tensor
template <typename TensorType, int SubDim, template <typename U> class PtrTraits>
class SubTensor {
public:
    // Returns a view of the data located at our offset (the dimension
    // `SubDim` - 1 tensor).
    inline SubTensor<TensorType, SubDim - 1, PtrTraits> operator[](typename TensorType::IndexType index)
    {
        if (SubDim == 1) {
            return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor, dataPtr + index);
        } else {
            return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor, dataPtr + static_cast<size_t>(index) *
                                                                                      tensor.getStride(TensorType::NUM_DIM -
                                                                                                       SubDim));
        }
    }

    // Returns a view of the data located at our offset (the dimension
    // `SubDim` - 1 tensor)(const).
    inline const SubTensor<TensorType, SubDim - 1, PtrTraits> operator[](typename TensorType::IndexType index) const
    {
        if (SubDim == 1) {
            return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor, dataPtr + index);
        } else {
            return SubTensor<TensorType, SubDim - 1, PtrTraits>(tensor, dataPtr + static_cast<size_t>(index) *
                                                                                      tensor.getStride(TensorType::NUM_DIM -
                                                                                                       SubDim));
        }
    }

    // operator& returning T*
    typename TensorType::DataType *operator&() { return dataPtr; }

    // const operator& returning const T*
    const typename TensorType::DataType *operator&() const { return dataPtr; }

    // Returns a raw accessor to our slice.
    typename TensorType::DataPtrType data() { return dataPtr; }

    // Returns a raw accessor to our slice (const).
    const typename TensorType::DataPtrType data() const { return dataPtr; }

    // Cast to a different datatype.
    template <typename T>
    T &as() { return *dataAs<T>(); }

    // Cast to a different datatype(const).
    template <typename T>
    const T &as() const { return *dataAs<T>(); }

    // Cast to a different datatype
    template <typename T>
    typename PtrTraits<T>::PtrType dataAs() { return reinterpret_cast<typename PtrTraits<T>::PtrType>(dataPtr); }

    // Cast to a different datatype(const)
    template <typename T>
    typename PtrTraits<const T>::PtrType dataAs() const { return reinterpret_cast<typename PtrTraits<const T>::PtrType>(dataPtr); }

    AscendTensor<typename TensorType::DataType, SubDim, typename TensorType::IndexType, PtrTraits> view() { return tensor.template view<SubDim>(dataPtr); }

    // defautl destructor function
    ~SubTensor() {}

protected:
    // One dimension greater can create us
    friend class SubTensor<TensorType, SubDim + 1, PtrTraits>;

    // / Our parent tensor can create us
    friend class AscendTensor<typename TensorType::DataType, TensorType::NUM_DIM, typename TensorType::IndexType,
                              PtrTraits>;

    SubTensor(TensorType &t, typename TensorType::DataPtrType data) : tensor(t), dataPtr(data) {}

    // The tensor we're referencing
    TensorType &tensor;

    // Where our value is located
    typename TensorType::DataPtrType const dataPtr;
};
}  // namespace detail

}  // namespace ascendSearchacc

#endif  // ASCEND_TENSOR_INCLUDED
