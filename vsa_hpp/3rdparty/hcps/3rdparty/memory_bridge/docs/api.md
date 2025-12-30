<p id="目录"></p>

# 目录
* [ock::hmm::OckHmmAsyncResultBase](#ock::hmm::OckHmmAsyncResultBase)
    * [OckHmmAsyncResultBase::WaitResult](#OckHmmAsyncResultBase::WaitResult)
        * [OckHmmAsyncResultBase::WaitResult入参说明](#OckHmmAsyncResultBase::WaitResult入参说明)
        * [OckHmmAsyncResultBase::WaitResult返回值说明](#OckHmmAsyncResultBase::WaitResult返回值说明) 
    * [OckHmmAsyncResultBase::Cancel](#OckHmmAsyncResultBase::Cancel)
        * [OckHmmAsyncResultBase::Cancel入参说明](#OckHmmAsyncResultBase::Cancel入参说明)
        * [OckHmmAsyncResultBase::Cancel返回值说明](#OckHmmAsyncResultBase::Cancel返回值说明)
* [ock::hmm::OckHmmResultBase](#ock::hmm::OckHmmResultBase)
    * [OckHmmResultBase::ErrorCode](#OckHmmResultBase::ErrorCode)
        * [OckHmmResultBase::ErrorCode入参说明](#OckHmmResultBase::ErrorCode入参说明)
        * [OckHmmResultBase::ErrorCode返回值说明](#OckHmmResultBase::ErrorCode返回值说明)
* [ock::hmm::OckHmmHeteroMemoryLocation](#ock::hmm::OckHmmHeteroMemoryLocation)
    * [LOCAL_HOST_MEMORY](#LOCAL_HOST_MEMORY)
    * [DEVICE_HBM](#DEVICE_HBM)
    * [DEVICE_DDR](#DEVICE_DDR)
* [ock::hmm::OckHmmMemoryAllocatePolicy](#ock::hmm::OckHmmMemoryAllocatePolicy)
    * [DEVICE_DDR_FIRST](#DEVICE_DDR_FIRST)
    * [DEVICE_DDR_ONLY](#DEVICE_DDR_ONLY)
    * [LOCAL_HOST_ONLY](#LOCAL_HOST_ONLY)
* [ock::hmm::OckHmmHeteroMemoryMgrBase](#ock::hmm::OckHmmHeteroMemoryMgrBase)
    * [OckHmmHeteroMemoryMgrBase::Alloc](#OckHmmHeteroMemoryMgrBase::Alloc)
        * [OckHmmHeteroMemoryMgrBase::Alloc入参说明](#OckHmmHeteroMemoryMgrBase::Alloc入参说明)
        * [OckHmmHeteroMemoryMgrBase::Alloc返回值说明](#OckHmmHeteroMemoryMgrBase::Alloc返回值说明)
    * [OckHmmHeteroMemoryMgrBase::Free](#OckHmmHeteroMemoryMgrBase::Free)
        * [OckHmmHeteroMemoryMgrBase::Free入参说明](#OckHmmHeteroMemoryMgrBase::Free入参说明)
        * [OckHmmHeteroMemoryMgrBase::Free返回值说明](#OckHmmHeteroMemoryMgrBase::Free返回值说明)
    * [OckHmmHeteroMemoryMgrBase::CopyHMO](#OckHmmHeteroMemoryMgrBase::CopyHMO)
        * [OckHmmHeteroMemoryMgrBase::CopyHMO入参说明](#OckHmmHeteroMemoryMgrBase::CopyHMO入参说明)
        * [OckHmmHeteroMemoryMgrBase::CopyHMO返回值说明](#OckHmmHeteroMemoryMgrBase::CopyHMO返回值说明)
    * [OckHmmHeteroMemoryMgrBase::GetUsedInfo](#OckHmmHeteroMemoryMgrBase::GetUsedInfo)
        * [OckHmmHeteroMemoryMgrBase::GetUsedInfo入参说明](#OckHmmHeteroMemoryMgrBase::GetUsedInfo入参说明)
        * [OckHmmHeteroMemoryMgrBase::GetUsedInfo返回值说明](#OckHmmHeteroMemoryMgrBase::GetUsedInfo返回值说明)
    * [OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo](#OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo)
        * [OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明](#OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明)
        * [OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明](#OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明)
* [ock::hmm::OckHmmSingleDeviceMgr](#ock::hmm::OckHmmSingleDeviceMgr)
    * [OckHmmSingleDeviceMgr::GetSpecific](#OckHmmSingleDeviceMgr::GetSpecific)
        * [OckHmmSingleDeviceMgr::GetSpecific入参说明](#OckHmmSingleDeviceMgr::GetSpecific入参说明)
        * [OckHmmSingleDeviceMgr::GetSpecific返回值说明](#OckHmmSingleDeviceMgr::GetSpecific返回值说明)
    * [OckHmmSingleDeviceMgr::GetDeviceId](#OckHmmSingleDeviceMgr::GetDeviceId)
        * [OckHmmSingleDeviceMgr::GetDeviceId入参说明](#OckHmmSingleDeviceMgr::GetDeviceId入参说明)
        * [OckHmmSingleDeviceMgr::GetDeviceId返回值说明](#OckHmmSingleDeviceMgr::GetDeviceId返回值说明)
    * [OckHmmSingleDeviceMgr::GetCpuSet](#OckHmmSingleDeviceMgr::GetCpuSet)
        * [OckHmmSingleDeviceMgr::GetCpuSet入参说明](#OckHmmSingleDeviceMgr::GetCpuSet入参说明)
        * [OckHmmSingleDeviceMgr::GetCpuSet返回值说明](#OckHmmSingleDeviceMgr::GetCpuSet返回值说明)
* [ock::hmm::OckHmmMultiDeviceHeteroMemoryMgrBase](#ock::hmm::OckHmmMultiDeviceHeteroMemoryMgrBase)
    * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo入参说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo入参说明)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo返回值说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo返回值说明)
    * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明)
    * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet入参说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet入参说明)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet返回值说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet返回值说明)
    * [OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc](#OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc入参说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc入参说明)
        * [OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc返回值说明](#OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc返回值说明)
* [ock::hmm::OckHmmComposeDeviceMgr](#ock::hmm::OckHmmComposeDeviceMgr)
    * [OckHmmComposeDeviceMgr::GetSpecific](#OckHmmComposeDeviceMgr::GetSpecific)
        * [OckHmmComposeDeviceMgr::GetSpecific入参说明](#OckHmmComposeDeviceMgr::GetSpecific入参说明)
        * [OckHmmComposeDeviceMgr::GetSpecific返回值说明](#OckHmmComposeDeviceMgr::GetSpecific返回值说明)
* [ock::hmm::OckHmmHMOBuffer](#ock::hmm::OckHmmHMOBuffer)
    * [OckHmmHMOBuffer::Address](#OckHmmHMOBuffer::Address)
        * [OckHmmHMOBuffer::Address入参说明](#OckHmmHMOBuffer::Address入参说明)
        * [OckHmmHMOBuffer::Address返回值说明](#OckHmmHMOBuffer::Address返回值说明)
    * [OckHmmHMOBuffer::Size](#OckHmmHMOBuffer::Size)
        * [OckHmmHMOBuffer::Size入参说明](#OckHmmHMOBuffer::Size入参说明)
        * [OckHmmHMOBuffer::Size返回值说明](#OckHmmHMOBuffer::Size返回值说明)
    * [OckHmmHMOBuffer::Offset](#OckHmmHMOBuffer::Offset)
        * [OckHmmHMOBuffer::Offset入参说明](#OckHmmHMOBuffer::Offset入参说明)
        * [OckHmmHMOBuffer::Offset返回值说明](#OckHmmHMOBuffer::Offset返回值说明)
    * [OckHmmHMOBuffer::Location](#OckHmmHMOBuffer::Location)
        * [OckHmmHMOBuffer::Location入参说明](#OckHmmHMOBuffer::Location入参说明)
        * [OckHmmHMOBuffer::Location返回值说明](#OckHmmHMOBuffer::Location返回值说明)
    * [OckHmmHMOBuffer::GetId](#OckHmmHMOBuffer::GetId)
        * [OckHmmHMOBuffer::GetId入参说明](#OckHmmHMOBuffer::GetId入参说明)
        * [OckHmmHMOBuffer::GetId返回值说明](#OckHmmHMOBuffer::GetId返回值说明)
    * [OckHmmHMOBuffer::FlushData](#OckHmmHMOBuffer::FlushData)
        * [OckHmmHMOBuffer::FlushData入参说明](#OckHmmHMOBuffer::FlushData入参说明)
        * [FlushData返回值说明](#FlushData返回值说明)
* [ock::hmm::OckHmmHMObject](#ock::hmm::OckHmmHMObject)
    * [OckHmmHMObject::GetId](#OckHmmHMObject::GetId)
        * [OckHmmHMObject::GetId入参说明](#OckHmmHMObject::GetId入参说明)
        * [OckHmmHMObject::GetId返回值说明](#OckHmmHMObject::GetId返回值说明)
    * [OckHmmHMObject::GetByteSize](#OckHmmHMObject::GetByteSize)
        * [OckHmmHMObject::GetByteSize入参说明](#OckHmmHMObject::GetByteSize入参说明)
        * [OckHmmHMObject::GetByteSize返回值说明](#OckHmmHMObject::GetByteSize返回值说明)
    * [OckHmmHMObject::Addr](#OckHmmHMObject::Addr)
        * [OckHmmHMObject::Addr入参说明](#OckHmmHMObject::Addr入参说明)
        * [OckHmmHMObject::Addr返回值说明](#OckHmmHMObject::Addr返回值说明)
    * [OckHmmHMObject::Location](#OckHmmHMObject::Location)
        * [OckHmmHMObject::Location入参说明](#OckHmmHMObject::Location入参说明)
        * [OckHmmHMObject::Location返回值说明](#OckHmmHMObject::Location返回值说明)
    * [OckHmmHMObject::GetBuffer](#OckHmmHMObject::GetBuffer)
        * [OckHmmHMObject::GetBuffer入参说明](#OckHmmHMObject::GetBuffer入参说明)
        * [OckHmmHMObject::GetBuffer返回值说明](#OckHmmHMObject::GetBuffer返回值说明)
    * [OckHmmHMObject::GetBufferAsync](#OckHmmHMObject::GetBufferAsync)
        * [OckHmmHMObject::GetBufferAsync入参说明](#OckHmmHMObject::GetBufferAsync入参说明)
        * [OckHmmHMObject::GetBufferAsync返回值说明](#OckHmmHMObject::GetBufferAsync返回值说明)
    * [OckHmmHMObject::ReleaseBuffer](#OckHmmHMObject::ReleaseBuffer)
        * [OckHmmHMObject::ReleaseBuffer入参说明](#OckHmmHMObject::ReleaseBuffer入参说明)
        * [OckHmmHMObject::ReleaseBuffer返回值说明](#OckHmmHMObject::ReleaseBuffer返回值说明)
* [ock::hmm::OckHmmMemoryGuard](#ock::hmm::OckHmmMemoryGuard)
    * [OckHmmMemoryGuard::Addr](#OckHmmMemoryGuard::Addr)
        * [OckHmmMemoryGuard::Addr入参说明](#OckHmmMemoryGuard::Addr入参说明)
        * [OckHmmMemoryGuard::Addr返回值说明](#OckHmmMemoryGuard::Addr返回值说明)
    * [OckHmmMemoryGuard::Location](#OckHmmMemoryGuard::Location)
        * [OckHmmMemoryGuard::Location入参说明](#OckHmmMemoryGuard::Location入参说明)
        * [OckHmmMemoryGuard::Location返回值说明](#OckHmmMemoryGuard::Location返回值说明)
    * [OckHmmMemoryGuard::ByteSize](#OckHmmMemoryGuard::ByteSize)
        * [OckHmmMemoryGuard::ByteSize入参说明](#OckHmmMemoryGuard::ByteSize入参说明)
        * [OckHmmMemoryGuard::ByteSize返回值说明](#OckHmmMemoryGuard::ByteSize返回值说明)
* [ock::hmm::OckHmmMemoryPool](#ock::hmm::OckHmmMemoryPool)
    * [OckHmmMemoryPool::Malloc](#OckHmmMemoryPool::Malloc)
        * [OckHmmMemoryPool::Malloc入参说明](#OckHmmMemoryPool::Malloc入参说明)
        * [OckHmmMemoryPool::Malloc返回值说明](#OckHmmMemoryPool::Malloc返回值说明)
* [ock::hmm::OckHmmErrorCode](#ock::hmm::OckHmmErrorCode)
    * [返回码分段说明](#返回码分段说明)
    * [返回码列表](#返回码列表)
* [ock::hmm::OckHmmHMOObjectID](#ock::hmm::OckHmmHMOObjectID)
* [ock::hmm::OckHmmDeviceId](#ock::hmm::OckHmmDeviceId)
* [ock::hmm::OckHmmFactory](#ock::hmm::OckHmmFactory)
    * [OckHmmFactory::CreateSingleDeviceMemoryMgr](#OckHmmFactory::CreateSingleDeviceMemoryMgr)
        * [OckHmmFactory::CreateSingleDeviceMemoryMgr入参说明](#OckHmmFactory::CreateSingleDeviceMemoryMgr入参说明)
        * [OckHmmFactory::CreateSingleDeviceMemoryMgr返回值说明](#OckHmmFactory::CreateSingleDeviceMemoryMgr返回值说明)
    * [OckHmmFactory::CreateComposeMemoryMgr](#OckHmmFactory::CreateComposeMemoryMgr)
        * [OckHmmFactory::CreateComposeMemoryMgr入参说明](#OckHmmFactory::CreateComposeMemoryMgr入参说明)
        * [OckHmmFactory::CreateComposeMemoryMgr返回值说明](#OckHmmFactory::CreateComposeMemoryMgr返回值说明)
    * [OckHmmFactory::Create](#OckHmmFactory::Create)
        * [OckHmmFactory::Create入参说明](#OckHmmFactory::Create入参说明)
        * [OckHmmFactory::Create返回值说明](#OckHmmFactory::Create返回值说明)

<p id="接口详细描述"></p>

# 接口详细描述
<p id="ock::hmm::OckHmmAsyncResultBase"></p>

## ock::hmm::OckHmmAsyncResultBase
* OckHmmAsyncResult.h
<p id="OckHmmAsyncResultBase::WaitResult"></p>

### OckHmmAsyncResultBase::WaitResult
```c++
virtual std::shared_ptr<_ResultT> WaitResult(uint32_t timeout) = 0;
```
<p id="OckHmmAsyncResultBase::WaitResult入参说明"></p>

#### OckHmmAsyncResultBase::WaitResult入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|timeout|uint32_t|{最大值: 4294967295, 最小值: 0}||
<p id="OckHmmAsyncResultBase::WaitResult返回值说明"></p>

#### OckHmmAsyncResultBase::WaitResult返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::shared_ptr<_ResultT>|||
<p id="ock::hmm::OckHmmResultBase"></p>

### OckHmmAsyncResultBase::Cancel
```c++
virtual void Cancel(void) = 0;
```
<p id="OckHmmAsyncResultBase::Cancel入参说明"></p>

#### OckHmmAsyncResultBase::Cancel入参说明
无
<p id="OckHmmAsyncResultBase::WaitResult返回值说明"></p>

#### OckHmmAsyncResultBase::Cancel返回值说明
无
<p id="ock::hmm::OckHmmResultBase"></p>

## ock::hmm::OckHmmResultBase
* OckHmmAsyncResult.h
<p id="OckHmmResultBase::ErrorCode"></p>

### OckHmmResultBase::ErrorCode
```c++
virtual ock::hmm::OckHmmErrorCode ErrorCode(void) = 0;
```
<p id="OckHmmResultBase::ErrorCode入参说明"></p>

#### OckHmmResultBase::ErrorCode入参说明
无
<p id="OckHmmResultBase::ErrorCode返回值说明"></p>

#### OckHmmResultBase::ErrorCode返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmErrorCode](#ock::hmm::OckHmmErrorCode)|||
<p id="ock::hmm::OckHmmHeteroMemoryLocation"></p>

## ock::hmm::OckHmmHeteroMemoryLocation
* OckHmmHeteroMemoryLocation.h
<p id="LOCAL_HOST_MEMORY"></p>

### LOCAL_HOST_MEMORY
本机内存。
<p id="DEVICE_HBM"></p>

### DEVICE_HBM
卡上HBM内存。
<p id="DEVICE_DDR"></p>

### DEVICE_DDR
卡上DDR内存。
<p id="ock::hmm::OckHmmMemoryAllocatePolicy"></p>

## ock::hmm::OckHmmMemoryAllocatePolicy
* OckHmmHeteroMemoryLocation.h
<p id="DEVICE_DDR_FIRST"></p>

### DEVICE_DDR_FIRST
<p id="DEVICE_DDR_ONLY"></p>

### DEVICE_DDR_ONLY
<p id="LOCAL_HOST_ONLY"></p>

### LOCAL_HOST_ONLY
<p id="ock::hmm::OckHmmHeteroMemoryMgrBase"></p>

## ock::hmm::OckHmmHeteroMemoryMgrBase
```c++
class OckHmmHeteroMemoryMgrBase : public OckHmmMemoryPool {
...
};
```
* 父类[OckHmmMemoryPool](#ock::hmm::OckHmmMemoryPool)
* OckHmmHeteroMemoryMgr.h
<p id="OckHmmHeteroMemoryMgrBase::Alloc"></p>

### OckHmmHeteroMemoryMgrBase::Alloc
内存分配，这里是内存的二次分配，性能高于直接调用ACL接口
```c++
virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject> > Alloc(uint64_t hmoBytes,ock::hmm::OckHmmMemoryAllocatePolicy policy) = 0;
```
<p id="OckHmmHeteroMemoryMgrBase::Alloc入参说明"></p>

#### OckHmmHeteroMemoryMgrBase::Alloc入参说明
| 参数名 | 参数类型 | 参数约束                                                | 其他说明                                                                                                                                                                                                                                     |
|---|---|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|hmoBytes|uint64_t| {最大值: 137438953472(128GB), 最小值: 0MB（不包括0）}， 单位：Byte | HMM的内存分配按照1M对齐（向上取整），因此不足1M时，会导致内存浪费（如申请10.1M内存，会分配11M内存，有0.9M浪费）                                                                                                                                                                        |
|policy|[ock::hmm::OckHmmMemoryAllocatePolicy](#ock::hmm::OckHmmMemoryAllocatePolicy)| 参考枚举[定义](#ock::hmm::OckHmmMemoryAllocatePolicy)     | <font color=cyan>DEVICE_DDR_FIRST</font> 优先在DEVICE上申请内存， 当Device内存不够时， 从HOST上申请内存，Device与HOST内存均不够时返回错误。<font color=cyan>DEVICE_DDR_ONLY</font> 只从DEVICE上申请，当内存不够时，返回错误。<font color=cyan>LOCAL_HOST_ONLY</font> 只从HOST上申请内存， 当内存不够时，返回错误 |
<p id="OckHmmHeteroMemoryMgrBase::Alloc返回值说明"></p>

#### OckHmmHeteroMemoryMgrBase::Alloc返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::pair<[OckHmmErrorCode](#ock::hmm::OckHmmErrorCode), std::shared_ptr<[OckHmmHMObject](#ock::hmm::OckHmmHMObject)> >|当返回结果.first值不为HMM_SUCCESS时，返回失败，此时返回结果的.second不可信。|一个设备的异构内存管理对象最多支持10万个，超过数目会返回错误|
<p id="OckHmmHeteroMemoryMgrBase::Free"></p>

### OckHmmHeteroMemoryMgrBase::Free
HMO对象释放，即使不主动调用Free接口， 用户拥有的HMO对象析构时，也会自动触发Free。
```c++
virtual void Free(std::shared_ptr<OckHmmHMObject> hmo) = 0;
```
<p id="OckHmmHeteroMemoryMgrBase::Free入参说明"></p>

#### OckHmmHeteroMemoryMgrBase::Free入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|hmo|std::shared_ptr<[OckHmmHMObject](#ock::hmm::OckHmmHMObject)>|1) 传入对象必须为[Alloc](#OckHmmHeteroMemoryMgrBase::Alloc)接口返回的对象。2) 传入的HMO为本管理对象实例分配 3) 不要反复Free相同对象(极限情况下可能释放错误) |1)为空指针时： 直接返回 2) 传入的HMO如果为非本管理对象实例分配，直接返回 3) 传入对象的objectId不合法，直接返回|
<p id="OckHmmHeteroMemoryMgrBase::Free返回值说明"></p>

#### OckHmmHeteroMemoryMgrBase::Free返回值说明
void
<p id="OckHmmHeteroMemoryMgrBase::CopyHMO"></p>

### OckHmmHeteroMemoryMgrBase::CopyHMO
HMO间的数据拷贝。<font color=red>暂不支持两个不同Device上的数据的拷贝</font>
* 当srcHMO与dstHMO的deviceId不相同是，返回不成功。
```c++
virtual ock::hmm::OckHmmErrorCode CopyHMO(ock::hmm::OckHmmHMObject & dstHMO,uint64_t dstOffset,ock::hmm::OckHmmHMObject & srcHMO,uint64_t srcOffset,size_t length) = 0;
```
<p id="OckHmmHeteroMemoryMgrBase::CopyHMO入参说明"></p>

#### OckHmmHeteroMemoryMgrBase::CopyHMO入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|dstHMO|[ock::hmm::OckHmmHMObject](#ock::hmm::OckHmmHMObject) &|1) 传入对象必须为[Alloc](#OckHmmHeteroMemoryMgrBase::Alloc)接口返回的对象。2) 传入的HMO为本管理对象实例分配 3) 传入对象未被Free ||
|dstOffset|uint64_t|1) dstOffset必须小于dstHMO.Size() 2) dstOffset + length小于等于dstHMO.Size()||
|srcHMO|[ock::hmm::OckHmmHMObject](#ock::hmm::OckHmmHMObject) &|1) 传入对象必须为[Alloc](#OckHmmHeteroMemoryMgrBase::Alloc)接口返回的对象。2) 传入的HMO为本管理对象实例分配 3) 传入对象未被Free||
|srcOffset|uint64_t|1) srcOffset必须小于srcHMO.Size() 2) srcOffset + length小于等于srcHMO.Size()||
|length|size_t|大于等于1|当length等于0时，直接返回成功|
<p id="OckHmmHeteroMemoryMgrBase::CopyHMO返回值说明"></p>

#### OckHmmHeteroMemoryMgrBase::CopyHMO返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmErrorCode](#ock::hmm::OckHmmErrorCode)|||
<p id="OckHmmHeteroMemoryMgrBase::GetUsedInfo"></p>

### OckHmmHeteroMemoryMgrBase::GetUsedInfo
获取内存使用信息
```c++
virtual std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(uint64_t fragThreshold) = 0;
```
<p id="OckHmmHeteroMemoryMgrBase::GetUsedInfo入参说明"></p>

#### OckHmmHeteroMemoryMgrBase::GetUsedInfo入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|fragThreshold|uint64_t|{最大值: 4371376400(4G), 最小值: 2121440(2M)}|连续空间小于fragThreshold的内存块认为是碎片，因为内存分配粒度而浪费的内存大小也算碎片。<font color=cyan>避免在高性能场景频繁调用查询接口</font> 当fragThreshold小于最小值时, 返回空指针|
<p id="OckHmmHeteroMemoryMgrBase::GetUsedInfo返回值说明"></p>

#### OckHmmHeteroMemoryMgrBase::GetUsedInfo返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::shared_ptr<[OckHmmResourceUsedInfo](#ock::hmm::OckHmmResourceUsedInfo)>||<font color=cyan>避免在高性能场景频繁调用查询接口</font>|
<p id="OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo"></p>

### OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo
获取流量统计信息
```c++
virtual std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(uint32_t maxGapMilliSeconds = 10) = 0;
```
<p id="OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明"></p>

#### OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明
| 参数名 | 参数类型 | 参数约束                                | 其他说明                                       |
|---|---|-------------------------------------|--------------------------------------------|
|maxGapMilliSeconds|uint32_t| {最大值: 1000, 最小值: 1}         | 缺省值：10                                     |
<p id="OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明"></p>

#### OckHmmHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明
| 返回值类型 | 返回值约束                                                                                           | 其他说明 |
|---|-------------------------------------------------------------------------------------------------|---|
|std::shared_ptr<[OckHmmTrafficStatisticsInfo](#ock::hmm::OckHmmTrafficStatisticsInfo)>| 统计数据跨度约束: 1）统计最近一次连续数据(两次传输命令间隔小于maxGapMilliSeconds(单位：ms))) 传输数据(不含命令传输) 2) 当传输数据量不为0时，传输速度才有效 |传输速度单位: MB/s|
<p id="ock::hmm::OckHmmSingleDeviceMgr"></p>

## ock::hmm::OckHmmSingleDeviceMgr
```c++
class OckHmmSingleDeviceMgr : public OckHmmHeteroMemoryMgrBase {
...
};
```
* 父类[OckHmmHeteroMemoryMgrBase](#ock::hmm::OckHmmHeteroMemoryMgrBase)
* OckHmmHeteroMemoryMgr.h
<p id="OckHmmSingleDeviceMgr::GetSpecific"></p>

### OckHmmSingleDeviceMgr::GetSpecific
获取内存规格信息
```c++
virtual const OckHmmMemorySpecification &GetSpecific(void) const = 0;
```
<p id="OckHmmSingleDeviceMgr::GetSpecific入参说明"></p>

#### OckHmmSingleDeviceMgr::GetSpecific入参说明
无
<p id="OckHmmSingleDeviceMgr::GetSpecific返回值说明"></p>

#### OckHmmSingleDeviceMgr::GetSpecific返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[OckHmmMemorySpecification](#ock::hmm::OckHmmMemorySpecification)||与管理对象创建时传入的内存规格对象的数值内容一致|
<p id="OckHmmSingleDeviceMgr::GetDeviceId"></p>

### OckHmmSingleDeviceMgr::GetDeviceId
获取Device的ID号
```c++
virtual OckHmmDeviceId GetDeviceId(void) const = 0;
```
<p id="OckHmmSingleDeviceMgr::GetDeviceId入参说明"></p>

#### OckHmmSingleDeviceMgr::GetDeviceId入参说明
无
<p id="OckHmmSingleDeviceMgr::GetDeviceId返回值说明"></p>

#### OckHmmSingleDeviceMgr::GetDeviceId返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[OckHmmDeviceId](#ock::hmm::OckHmmDeviceId)||与管理对象创建时传入的DeviceId一致|
<p id="OckHmmSingleDeviceMgr::GetCpuSet"></p>

### OckHmmSingleDeviceMgr::GetCpuSet
```c++
virtual const cpu_set_t &GetCpuSet(void) const = 0;
```
<p id="OckHmmSingleDeviceMgr::GetCpuSet入参说明"></p>

#### OckHmmSingleDeviceMgr::GetCpuSet入参说明
无
<p id="OckHmmSingleDeviceMgr::GetCpuSet返回值说明"></p>

#### OckHmmSingleDeviceMgr::GetCpuSet返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|cpu_set_t||与管理对象创建时传入的cpu_set_t的数值内容一致|

<p id="ock::hmm::OckHmmMultiDeviceHeteroMemoryMgrBase"></p>

## ock::hmm::OckHmmMultiDeviceHeteroMemoryMgrBase
* OckHmmHeteroMemoryMgr.h
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo"></p>

### OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo
获取内存使用信息
```c++
virtual std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(uint64_t fragThreshold,ock::hmm::OckHmmDeviceId deviceId) = 0;
```
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo入参说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|fragThreshold|uint64_t|{最大值: 4371376400(4G), 最小值: 2121440(2M)}|连续空间小于fragThreshold的内存块认为是碎片，因为内存分配粒度而浪费的内存大小也算碎片。<font color=cyan>避免在高性能场景频繁调用查询接口</font> 当fragThreshold小于最小值时, 返回空指针。当传入的deviceId不在本管理对象时，返回shared_ptr的管理对象为空指针|
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo返回值说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::GetUsedInfo返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::shared_ptr<[OckHmmResourceUsedInfo](#ock::hmm::OckHmmResourceUsedInfo)>|||
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo"></p>

### OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo
获取流量统计信息
```c++
virtual std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(ock::hmm::OckHmmDeviceId deviceId, uint32_t maxGapMilliSeconds = 10) = 0;
```
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo入参说明
| 参数名 | 参数类型 | 参数约束                                | 其他说明                                       |
|---|---|-------------------------------------|--------------------------------------------|
|deviceId|ock::hmm::OckHmmDeviceId| {最大值: 18446744073709551615, 最小值: 0} | 当传入的deviceId不在本管理对象时，返回shared_ptr的管理对象为空指针 |
|maxGapMilliSeconds|uint32_t| {最大值: 1000, 最小值: 1}         | 缺省值：10                                     |
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::GetTrafficStatisticsInfo返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::shared_ptr<[OckHmmTrafficStatisticsInfo](#ock::hmm::OckHmmTrafficStatisticsInfo)>|||
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet"></p>

### OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet
获取cpu_set_t信息 HMM模块内部线程都绑定在对应的cpu_set_t上
```c++
const cpu_set_t *GetCpuSet(ock::hmm::OckHmmDeviceId deviceId);
```
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet入参说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|deviceId|ock::hmm::OckHmmDeviceId|{最大值: 18446744073709551615, 最小值: 0}|当传入的deviceId不在本管理对象时，返回空指针|
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet返回值说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::GetCpuSet返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|const cpu_set_t *|||
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc"></p>

### OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc
在指定deviceId的单卡异构内存管理对象中分配HMO
```c++
virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(OckHmmDeviceId deviceId, uint64_t hmoBytes,
    OckHmmMemoryAllocatePolicy policy = OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) = 0;
```
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc入参说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc入参说明
| 参数名 | 参数类型 | 参数约束                                                 | 其他说明 |
|---|---|------------------------------------------------------|---|
|deviceId|ock::hmm::OckHmmDeviceId| {最大值: 18446744073709551615, 最小值: 0}                  |传入的deviceId必须在本管理对象内|
|hmoBytes|uint64_t| {最大值: 137438953472(128GB), 最小值: 0MB（不包括0）}, 单位: byte |HMM的内存分配按照1MB对齐，因此不足1MB时，会造成内存浪费|
|policy|[ock::hmm::OckHmmMemoryAllocatePolicy](#ock::hmm::OckHmmMemoryAllocatePolicy)| 参考枚举定义                                               |DEVICE_DDR_FIRST优先在device上申请内存，当device内存不够时，从host上申请，二者内存均不足时返回错误。DEVICE_DDR_ONLY只从device上申请内存，内存不够时，返回错误。LOCAL_HOST_ONLY只从host上申请内存，当内存不够时，返回错误。
<p id="OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc返回值说明"></p>

#### OckHmmMultiDeviceHeteroMemoryMgrBase::Alloc返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::pair<[ock::hmm::OckHmmErrorCode](#ock::hmm::OckHmmErrorCode), std::shared_ptr<[ock::hmm::OckHmmHMObject](#ock::hmm::OckHmmHMObject)>>|当返回结果的.first值不为HMM_SUCCESS时，此时返回的结果的.second不可信|
<p id="ock::hmm::OckHmmComposeDeviceMgr"></p>

## ock::hmm::OckHmmComposeDeviceMgr
```c++
class OckHmmComposeDeviceMgr : public OckHmmMultiDeviceHeteroMemoryMgrBase, public OckHmmHeteroMemoryMgrBase {
...
};
```
* 父类[OckHmmMultiDeviceHeteroMemoryMgrBase](#ock::hmm::OckHmmMultiDeviceHeteroMemoryMgrBase)
* 父类[OckHmmHeteroMemoryMgrBase](#ock::hmm::OckHmmHeteroMemoryMgrBase)
* OckHmmHeteroMemoryMgr.h
<p id="OckHmmComposeDeviceMgr::GetSpecific"></p>

### OckHmmComposeDeviceMgr::GetSpecific
获取指定设备的内存规格信息
```c++
virtual const OckHmmMemorySpecification *GetSpecific(ock::hmm::OckHmmDeviceId deviceId) const = 0;
```
<p id="OckHmmComposeDeviceMgr::GetSpecific入参说明"></p>

#### OckHmmComposeDeviceMgr::GetSpecific入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|deviceId|ock::hmm::OckHmmDeviceId|传入的DeviceId属于本管理对象关联的DeviceID||
<p id="OckHmmComposeDeviceMgr::GetSpecific返回值说明"></p>

#### OckHmmComposeDeviceMgr::GetSpecific返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|const [OckHmmMemorySpecification](#ock::hmm::OckHmmMemorySpecification) *||当传入的deviceId不在本管理对象时，返回空指针|
<p id="OckHmmComposeDeviceMgr::Alloc"></p>

### OckHmmComposeDeviceMgr::Alloc
指定设备ID做内存分配，这里是内存的二次分配，性能高于直接调用ACL接口
```c++
virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject> > Alloc(ock::hmm::OckHmmDeviceId deviceId, uint64_t hmoBytes,ock::hmm::OckHmmMemoryAllocatePolicy policy) = 0;
```
<p id="OckHmmComposeDeviceMgr::Alloc入参说明"></p>

#### OckHmmComposeDeviceMgr::Alloc入参说明
| 参数名 | 参数类型 | 参数约束                                            | 其他说明 |
|---|---|-------------------------------------------------|---|
|deviceId|ock::hmm::OckHmmDeviceId| 传入的DeviceId属于本管理对象关联的DeviceID                   ||
|hmoBytes|uint64_t| {最大值: 4294967296(4GB), 最小值: 0MB（不包括0）}， 单位：Byte |HMM的内存分配按照2M对齐，因此不足2M时，会导致内存浪费|
|policy|[ock::hmm::OckHmmMemoryAllocatePolicy](#ock::hmm::OckHmmMemoryAllocatePolicy)| 参加枚举[定义](#ock::hmm::OckHmmMemoryAllocatePolicy) |<font color=cyan>DEVICE_DDR_FIRST</font> 优先在DEVICE上申请内存， 当Device内存不够时， 从HOST上申请内存，Device与HOST内存均不够时返回错误。<font color=cyan>DEVICE_DDR_ONLY</font> 只从DEVICE上申请，当内存不够时，返回错误。<font color=cyan>LOCAL_HOST_ONLY</font> 只从HOST上申请内存， 当内存不够时，返回错误|
<p id="OckHmmComposeDeviceMgr::Alloc返回值说明"></p>

#### OckHmmComposeDeviceMgr::Alloc返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::pair<[OckHmmErrorCode](#ock::hmm::OckHmmErrorCode), std::shared_ptr<[OckHmmHMObject](#ock::hmm::OckHmmHMObject)> >|当返回结果.first值不为HMM_SUCCESS时，返回失败，此时返回结果的.second不可信。|一个设备的异构内存管理对象最多支持10万个，超过数目会返回错误|
<p id="ock::hmm::OckHmmHMOBuffer"></p>

## ock::hmm::OckHmmHMOBuffer
* OckHmmHMObject.h
<p id="OckHmmHMOBuffer::Address"></p>

### OckHmmHMOBuffer::Address
该Buffer地址
```c++
virtual uintptr_t Address(void) = 0;
```
<p id="OckHmmHMOBuffer::Address入参说明"></p>

#### OckHmmHMOBuffer::Address入参说明
无
<p id="OckHmmHMOBuffer::Address返回值说明"></p>

#### OckHmmHMOBuffer::Address返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uintptr_t|当返回值为0时，地址不合法。当OckHmmHMOBuffer对象所指向的内存被回收时，返回值为0|HMO对象所在位置的地址|
<p id="OckHmmHMOBuffer::Size"></p>

### OckHmmHMOBuffer::Size
该Buffer大小
```c++
virtual uint64_t Size(void) = 0;
```
<p id="OckHmmHMOBuffer::Size入参说明"></p>

#### OckHmmHMOBuffer::Size入参说明
无
<p id="OckHmmHMOBuffer::Size返回值说明"></p>

#### OckHmmHMOBuffer::Size返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uint64_t||当OckHmmHMOBuffer对象所指向的内存被回收时，返回值为0|
<p id="OckHmmHMOBuffer::Offset"></p>

### OckHmmHMOBuffer::Offset
该Buffer相对对应HMO的Offset位置
```c++
virtual uint64_t Offset(void) = 0;
```
<p id="OckHmmHMOBuffer::Offset入参说明"></p>

#### OckHmmHMOBuffer::Offset入参说明
无
<p id="OckHmmHMOBuffer::Offset返回值说明"></p>

#### OckHmmHMOBuffer::Offset返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uint64_t||当OckHmmHMOBuffer对象所指向的内存被回收时，返回值为0|
<p id="OckHmmHMOBuffer::Location"></p>

### OckHmmHMOBuffer::Location
该Buffer的位置类型
```c++
virtual ock::hmm::OckHmmHeteroMemoryLocation Location(void) = 0;
```
<p id="OckHmmHMOBuffer::Location入参说明"></p>

#### OckHmmHMOBuffer::Location入参说明
无
<p id="OckHmmHMOBuffer::Location返回值说明"></p>

#### OckHmmHMOBuffer::Location返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmHeteroMemoryLocation](#ock::hmm::OckHmmHeteroMemoryLocation)|||
<p id="OckHmmHMOBuffer::GetId"></p>

### OckHmmHMOBuffer::GetId
该Buffer所属的HMO对象ID
```c++
virtual ock::hmm::OckHmmHMOObjectID GetId(void) = 0;
```
<p id="OckHmmHMOBuffer::GetId入参说明"></p>

#### OckHmmHMOBuffer::GetId入参说明
无
<p id="OckHmmHMOBuffer::GetId返回值说明"></p>

#### OckHmmHMOBuffer::GetId返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmHMOObjectID](#ock::hmm::OckHmmHMOObjectID)|||
<p id="OckHmmHMOBuffer::FlushData"></p>

### OckHmmHMOBuffer::FlushData
将buffer空间的数据刷写到HMO
```c++
virtual ock::hmm::OckHmmErrorCode FlushData(void) = 0;
```
<p id="OckHmmHMOBuffer::FlushData入参说明"></p>

#### OckHmmHMOBuffer::FlushData入参说明
无
<p id="FlushData返回值说明"></p>

#### FlushData返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmErrorCode](#ock::hmm::OckHmmErrorCode)|||
<p id="ock::hmm::OckHmmHMObject"></p>

## ock::hmm::OckHmmHMObject
* OckHmmHMObject.h
<p id="OckHmmHMObject::GetId"></p>

### OckHmmHMObject::GetId
```c++
virtual ock::hmm::OckHmmHMOObjectID GetId(void) = 0;
```
<p id="OckHmmHMObject::GetId入参说明"></p>

#### OckHmmHMObject::GetId入参说明
无
<p id="OckHmmHMObject::GetId返回值说明"></p>

#### OckHmmHMObject::GetId返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmHMOObjectID](#ock::hmm::OckHmmHMOObjectID)|||
<p id="OckHmmHMObject::GetByteSize"></p>

### OckHmmHMObject::GetByteSize
```c++
virtual uint64_t GetByteSize(void) = 0;
```
<p id="OckHmmHMObject::GetByteSize入参说明"></p>

#### OckHmmHMObject::GetByteSize入参说明
无
<p id="OckHmmHMObject::GetByteSize返回值说明"></p>

#### OckHmmHMObject::GetByteSize返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uint64_t|{最大值: 18446744073709551615, 最小值: 0}||
<p id="OckHmmHMObject::Addr"></p>

### OckHmmHMObject::Addr
```c++
virtual uintptr_t Addr(void) = 0;
```
<p id="OckHmmHMObject::Addr入参说明"></p>

#### OckHmmHMObject::Addr入参说明
无
<p id="OckHmmHMObject::Addr返回值说明"></p>

#### OckHmmHMObject::Addr返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uintptr_t|||
<p id="OckHmmHMObject::Location"></p>

### OckHmmHMObject::Location
```c++
virtual ock::hmm::OckHmmHeteroMemoryLocation Location(void) = 0;
```
<p id="OckHmmHMObject::Location入参说明"></p>

#### OckHmmHMObject::Location入参说明
无
<p id="OckHmmHMObject::Location返回值说明"></p>

#### OckHmmHMObject::Location返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|[ock::hmm::OckHmmHeteroMemoryLocation](#ock::hmm::OckHmmHeteroMemoryLocation)|||
<p id="OckHmmHMObject::GetBuffer"></p>

### OckHmmHMObject::GetBuffer
获取buffer数据(同步接口)，根据指定的location获取本HMO的镜像数据。 当指定location与HMO的Location()相同时返回HMO的“引用”（不会触发数据复制)数据。当指定location与HMO的Location()不相同时返回HMO的复制数据。
```c++
virtual std::shared_ptr<OckHmmHMOBuffer> GetBuffer(ock::hmm::OckHmmHeteroMemoryLocation location,uint64_t offset,uint64_t length,uint32_t timeout=0) = 0;
```
<p id="OckHmmHMObject::GetBuffer入参说明"></p>

#### OckHmmHMObject::GetBuffer入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|location|[ock::hmm::OckHmmHeteroMemoryLocation](#ock::hmm::OckHmmHeteroMemoryLocation)|||
|offset|uint64_t|offset + length 小于等于HMO的Size()||
|length|uint64_t|1) length 小于等于HMO的Size() 2) 最小值：1 3) length小于等于相应位置的maxSwapCapacity大小||
|timeout|uint32_t|缺省值0(当为0时，取最大值)，最大值3600000(1小时)， 单位：毫秒||
<p id="OckHmmHMObject::GetBuffer返回值说明"></p>

#### OckHmmHMObject::GetBuffer返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::shared_ptr<[OckHmmHMOBuffer](#ock::hmm::OckHmmHMOBuffer)>||1) 同一个HMO对象，同一时间可以生成多个Buffer对象 2) 入参不正确时，返回的shared_ptr指向空指针 3) 用户可以通过主动调用[ReleaseBuffer](#ReleaseBuffer)或触发OckHmmHMOBuffer的析构完成Buffer指向的swap空间的回收 4) Buffer对象虽然由HMO对象申请，但其并不会随着HMO对象的消亡而消亡，HMO对象析构后，Buffer对象仍可以独立存在|
<p id="OckHmmHMObject::GetBufferAsync"></p>

### OckHmmHMObject::GetBufferAsync
获取buffer数据(异步接口), 根据指定的location获取本HMO的镜像数据。 当指定location与HMO的Location()相同时返回HMO的“引用”（不会触发数据复制)数据。当指定location与HMO的Location()不相同时返回HMO的复制数据。
```c++
virtual std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer> > GetBufferAsync(ock::hmm::OckHmmHeteroMemoryLocation location,uint64_t offset,uint64_t length) = 0;
```
<p id="OckHmmHMObject::GetBufferAsync入参说明"></p>

#### OckHmmHMObject::GetBufferAsync入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|location|[ock::hmm::OckHmmHeteroMemoryLocation](#ock::hmm::OckHmmHeteroMemoryLocation)|||
|offset|uint64_t|offset + length 小于等于HMO的Size()||
|length|uint64_t|1) length 小于等于HMO的Size() 2) 最小值：1 3) length小于等于相应位置的maxSwapCapacity大小||
<p id="OckHmmHMObject::GetBufferAsync返回值说明"></p>

#### OckHmmHMObject::GetBufferAsync返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer> >|1) 同一个HMO对象，同一时间可以生成多个Buffer对象。2) 入参不正确时，返回的shared_ptr指向空指针, 3) 用户可调用OckHmmAsyncResult的WaitResult获取Buffer结果||
<p id="OckHmmHMObject::ReleaseBuffer"></p>

### OckHmmHMObject::ReleaseBuffer
释放Buffer数据。
```c++
virtual void ReleaseBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer) = 0;
```
<p id="OckHmmHMObject::ReleaseBuffer入参说明"></p>

#### OckHmmHMObject::ReleaseBuffer入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|---|---|---|---|
|buffer|std::shared_ptr<[OckHmmHMOBuffer](#ock::hmm::OckHmmHMOBuffer)>|1. 传入的buffer对象必须是本HMO管理对象的[GetBufferAsync](#GetBufferAsync)的[WaitResult](#WaitResult)或[GetBuffer](#GetBuffer)接口获取的Buffer对象||
<p id="OckHmmHMObject::ReleaseBuffer返回值说明"></p>

#### OckHmmHMObject::ReleaseBuffer返回值说明
void
<p id="ock::hmm::OckHmmMemoryGuard"></p>

## ock::hmm::OckHmmMemoryGuard
* OckHmmMemoryPool.h
<p id="OckHmmMemoryGuard::Addr"></p>

### OckHmmMemoryGuard::Addr
守护的内存地址
```c++
virtual uintptr_t Addr(void) = 0;
```
<p id="OckHmmMemoryGuard::Addr入参说明"></p>

#### OckHmmMemoryGuard::Addr入参说明
无
<p id="OckHmmMemoryGuard::Addr返回值说明"></p>

#### OckHmmMemoryGuard::Addr返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uintptr_t|内存地址||
<p id="OckHmmMemoryGuard::Location"></p>

### OckHmmMemoryGuard::Location
内存所在的位置类别
```c++
virtual hmm::OckHmmHeteroMemoryLocation Location(void) = 0;
```
<p id="OckHmmMemoryGuard::Location入参说明"></p>

#### OckHmmMemoryGuard::Location入参说明
无
<p id="OckHmmMemoryGuard::Location返回值说明"></p>

#### OckHmmMemoryGuard::Location返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|hmm::OckHmmHeteroMemoryLocation|||
<p id="OckHmmMemoryGuard::ByteSize"></p>

### OckHmmMemoryGuard::ByteSize
内存大小
```c++
virtual uint64_t ByteSize(void) = 0;
```
<p id="OckHmmMemoryGuard::ByteSize入参说明"></p>

#### OckHmmMemoryGuard::ByteSize入参说明
无
<p id="OckHmmMemoryGuard::ByteSize返回值说明"></p>

#### OckHmmMemoryGuard::ByteSize返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|uint64_t|{最大值: 18446744073709551615, 最小值: 0}||
<p id="ock::hmm::OckHmmMemoryPool"></p>

## ock::hmm::OckHmmMemoryPool
* OckHmmMemoryPool.h
<p id="OckHmmMemoryPool::Malloc"></p>

### OckHmmMemoryPool::Malloc
```c++
std::unique_ptr<OckHmmMemoryGuard> Malloc(uint64_t size,ock::hmm::OckHmmMemoryAllocatePolicy policy);
```
<p id="OckHmmMemoryPool::Malloc入参说明"></p>

#### OckHmmMemoryPool::Malloc入参说明
| 参数名 | 参数类型 | 参数约束                                      | 其他说明 |
|---|---|-------------------------------------------|---|
|size|uint64_t| {最大值: 4294967296(4GB), 最小值: 0MB}， 单位：Byte |HMM的内存分配按照2M对齐，因此不足2M时，会导致内存浪费|
|policy|[ock::hmm::OckHmmMemoryAllocatePolicy](#ock::hmm::OckHmmMemoryAllocatePolicy)| 同                                         |<font color=cyan>DEVICE_DDR_FIRST</font> 优先在DEVICE上申请内存， 当Device内存不够时， 从HOST上申请内存，Device与HOST内存均不够时返回错误。<font color=cyan>DEVICE_DDR_ONLY</font> 只从DEVICE上申请，当内存不够时，返回错误。<font color=cyan>LOCAL_HOST_ONLY</font> 只从HOST上申请内存， 当内存不够时，返回错误|
<p id="OckHmmMemoryPool::Malloc返回值说明"></p>

#### OckHmmMemoryPool::Malloc返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|---|---|---|
|std::unique_ptr<OckHmmMemoryGuard>|随着OckHmmMemoryGuard的析构完成内存回收||

<p id="ock::hmm::OckHmmErrorCode"></p>

## ock::hmm::OckHmmErrorCode
* OckHmmErrorCode.h
<p id="返回码分段说明"></p>

### 返回码分段说明
<font color=cyan>数据范围说明</font>
* 0代表成功
* 0~255供可执行程序返回码使用
* 10000~20000 HMM 内部使用
* 100000~1000000 ACL模块使用
具体含义参见AscendCL API参考

<p id="返回码列表"></p>

### 返回码列表
|返回码|含义|可能原因及解决办法|
|---|---|---|
|constexpr OckHmmErrorCode HMM_SUCCESS = 0|成功|-|
|constexpr OckHmmErrorCode HMM_ERROR_EXEC_INVALID_INPUT_PARAM = 1|可执行程序入参错误|-|
|constexpr OckHmmErrorCode HMM_ERROR_EXEC_FAILED = 2|可执行程序运行错误|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_EMPTY = 14000|输入参数为空或输入参数列表为空|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE = 14001|输入的参数超出规定的取值范围|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS = 14002|输入的DEVICE不存在|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS = 14004|cpuSet中的某个cpuId不存在
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_SRC_OFFSET_EXCEED_SCOPE = 14010|源数据OFFSET超出范围|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE = 14011|目标数据OFFSET超出范围|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_SRC_LENGTH_EXCEED_SCOPE = 14012|源数据长度超出范围|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DST_LENGTH_EXCEED_SCOPE = 14013|目标数据长度超出范围|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL = 14014|输入DeviceId不匹配|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP = 14050|不支持这样的操作|-|
|constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_ZERO_MALLOC = 14051|分配0个字节|-|
|constexpr OckHmmErrorCode HMM_ERROR_WAIT_TIME_OUT = 14060|等待超时|-|
|constexpr OckHmmErrorCode HMM_ERROR_DEVICE_BUFFER_SPACE_NOT_ENOUGH = 14070|Device的BUFFER(Swap)空间不足|-|
|constexpr OckHmmErrorCode HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH = 14071|Device的底库空间不足|-|
|constexpr OckHmmErrorCode HMM_ERROR_HOST_BUFFER_SPACE_NOT_ENOUGH = 14072|Host的BUFFER(Swap)空间不足|-|
|constexpr OckHmmErrorCode HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH = 14073|Host的底库空间不足|-|
|constexpr OckHmmErrorCode HMM_ERROR_SPACE_NOT_ENOUGH = 14074|空间不足|-|
|constexpr OckHmmErrorCode HMM_ERROR_SWAP_SPACE_NOT_ENOUGH = 14075|sawp空间不足|-|
|constexpr OckHmmErrorCode HMM_ERROR_TASK_ALREADY_RUNNING = 14100|任务已经在运行|-|
|constexpr OckHmmErrorCode HMM_ERROR_HMO_NO_AVAIBLE_ID = 14200|无可用的HMOID|-|
|constexpr OckHmmErrorCode HMM_ERROR_HMO_OBJECT_NUM_EXCEED = 14201|HMO对象超过上限|-|
|constexpr OckHmmErrorCode HMM_ERROR_HMO_OBJECT_NOT_EXISTS = 14202|HMO对象不存在|-|
|constexpr OckHmmErrorCode HMM_ERROR_HMO_OBJECT_INVALID = 14203|HMO对象不合法|-|
|constexpr OckHmmErrorCode HMM_ERROR_HMO_BUFFER_RELEASED = 14204|buffer数据已经被释放|-|
|constexpr OckHmmErrorCode HMM_ERROR_HMO_BUFFER_NOT_ALLOCED = 14205|buffer数据未分配，不存在|-|
|constexpr OckHmmErrorCode HMM_ERROR_UNKOWN_INNER_ERROR = 19999|未知错误|-|

<p id="ock::hmm::OckHmmHMOObjectID"></p>

## ock::hmm::OckHmmHMOObjectID
* OckHmmHMObject.h
<p id="ock::hmm::OckHmmDeviceId"></p>

## ock::hmm::OckHmmDeviceId
* OckHmmHMObject.h

<p id="ock::hmm::OckHmmFactory"></p>

## ock::hmm::OckHmmFactory
* OckHmmFactory.h
<p id="OckHmmFactory::CreateSingleDeviceMemoryMgr"></p>

### OckHmmFactory::CreateSingleDeviceMemoryMgr
创建单个设备的内存管理对象
```c++
virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr> > CreateSingleDeviceMemoryMgr(std::shared_ptr<OckHmmDeviceInfo> deviceInfo, uint32_t timeout = MAX_CREATOR_TIMEOUT_MILLISECONDS) = 0;
```
<p id="OckHmmFactory::CreateSingleDeviceMemoryMgr入参说明"></p>

#### OckHmmFactory::CreateSingleDeviceMemoryMgr入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|--|--|--|--|
|deviceInfo|std::shared_ptr<OckHmmDeviceInfo>|1.传入数据不能为空||
|deviceInfo->deviceId|[OckHmmDeviceId](#ock::hmm::OckHmmDeviceId)|1. 有效范围[0,16) 2. deviceId对应的设备必须存在|使用前请手动赋值，否则deviceInfo无法使用|
|deviceInfo->cpuSet|cpu_set_t|被set的cpuId对应的物理核必须存在|在使用前需通过CPU_ZERO(&cpuSet)清空cpuSet中的全部内容。如需绑核，请使用CPU_SET(cpuId, &cpuSet)进行绑核操作；否则，在CPU_ZERO操作后直接使用。|
|deviceInfo->transferThreadNum|uint32_t|1. 有效范围[1, 8]|内部线程数根据transferThreadNum做优化，可能有1~2个线程数的偏差|使用前请手动赋值，否则deviceInfo无法使用|
|deviceInfo->memorySpec.devSpec.maxDataCapacity|uint64_t|1) 单位： 字节, 2) 最大值：512 * 1024*1024*1024(512G) 最小值：1024*1024*1024(1G) 3) 设备的maxDataCapacity+maxSwapCapacity大小要小于卡上可用内存大小+预留空间(本程序不检测)|使用前请手动赋值，否则deviceInfo无法使用|
|deviceInfo->memorySpec.devSpec.maxSwapCapacity|uint64_t|1) 单位： 字节, 2) 最大值：8 * 1024*1024*1024(8G) 最小值：1024*1024*64(64M) 3) 设备的maxDataCapacity+maxSwapCapacity大小要小于卡上可用内存大小+预留空间(本程序不检测)|使用前请手动赋值，否则deviceInfo无法使用|
|deviceInfo->memorySpec.hostSpec.maxDataCapacity|uint64_t|1) 单位： 字节, 2) 最大值：512 * 1024*1024*1024(512G) 最小值：1024*1024*1024(1G) 3) Host的maxDataCapacity+maxSwapCapacity大小要小于Host上可用内存大小+预留空间(本程序不检测)|使用前请手动赋值，否则deviceInfo无法使用|
|deviceInfo->memorySpec.hostSpec.maxSwapCapacity|uint64_t|1) 单位： 字节, 2) 最大值：8 * 1024*1024*1024(8G) 最小值：1024*1024*64(64M) 3) Host的maxDataCapacity+maxSwapCapacity大小要小于Host上可用内存大小+预留空间(本程序不检测)|使用前请手动赋值，否则deviceInfo无法使用|
|timeout|uint32_t|默认值为5000（单位：毫秒），根据maxBaseCapacity和maxSwapCapacity大小不同，可以设置不同的等待时间，0~32GB建议使用默认值，32~128GB建议设置为10000，128~256GB建议设置为20000,256GB~512GB建议设置为50000以上|参数约束中仅为建议值，实际应用中可以根据实际情况自行调整，等待时间范围是1~3600000ms，当入参为0或者大于3600000时，取最长等待时间|
<p id="OckHmmFactory::CreateSingleDeviceMemoryMgr返回值说明"></p>

#### OckHmmFactory::CreateSingleDeviceMemoryMgr返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|--|--|--|
|std::pair<[OckHmmErrorCode](#ock::hmm::OckHmmErrorCode), std::shared_ptr<[OckHmmSingleDeviceMgr](#ock::hmm::OckHmmSingleDeviceMgr)> >|1. 返回的OckHmmErrorCode不为HMM_SUCCESS时, pair的second结果不可信|入参不合法时， 返回pair的first值不为HMM_SUCCESS|
<p id="OckHmmFactory::CreateComposeMemoryMgr"></p>

### OckHmmFactory::CreateComposeMemoryMgr
创建多个设备的内存管理对象
```c++
std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr> > CreateComposeMemoryMgr(std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec, uint32_t timeout = MAX_CREATOR_TIMEOUT_MILLISECONDS);
```
<p id="OckHmmFactory::CreateComposeMemoryMgr入参说明"></p>

#### OckHmmFactory::CreateComposeMemoryMgr入参说明
| 参数名 | 参数类型 | 参数约束 | 其他说明 |
|--|--|--|--|
|deviceInfoVec|std::shared_ptr<OckHmmDeviceInfoVec>|1. 不能为空，2. OckHmmDeviceInfoVec中至少有1个对象|OckHmmDeviceInfoVecVec = std::vector<[OckHmmDeviceInfoVec](#ock::hmm::OckHmmDeviceInfoVec)>|
|timeout|uint32_t|默认值为5000（单位：毫秒），根据maxBaseCapacity和maxSwapCapacity大小不同，可以设置不同的等待时间，0~32GB建议使用默认值，32~128GB建议设置为10000，128~256GB建议设置为20000,256GB~512GB建议设置为50000以上|参数约束中仅为建议值，实际应用中可以根据实际情况自行调整，等待时间范围是1~3600000ms，当入参为0或者大于3600000时，取最长等待时间|
<p id="OckHmmFactory::CreateComposeMemoryMgr返回值说明"></p>

#### OckHmmFactory::CreateComposeMemoryMgr返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|--|--|--|
|std::pair<[OckHmmErrorCode](#ock::hmm::OckHmmErrorCode), std::shared_ptr<[OckHmmComposeDeviceMgr](#ock::hmm::OckHmmComposeDeviceMgr)> >|||
<p id="OckHmmFactory::Create"></p>

### OckHmmFactory::Create
```c++
std::shared_ptr<OckHmmFactory> Create(void);
```
<p id="OckHmmFactory::Create入参说明"></p>

#### OckHmmFactory::Create入参说明
无
<p id="OckHmmFactory::Create返回值说明"></p>

#### OckHmmFactory::Create返回值说明
| 返回值类型 | 返回值约束 | 其他说明 |
|--|--|--|
|std::shared_ptr<[OckHmmFactory](#ock::hmm::OckHmmFactory)>|||
