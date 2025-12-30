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

#ifndef ASCEND_INDEX_TS_INCLUDED
#define ASCEND_INDEX_TS_INCLUDED
#include <memory>
#include <mutex>

namespace ascend {
class TSSuperBase;
}

namespace faiss {
namespace ascend {

using APP_ERROR = int;
constexpr int TS_MAX_TOPK = 1e5;
constexpr int TS_MAX_SEARCH = 10240; //  限定count最大值为10240
constexpr uint64_t DEFAULT_RESOURCE_SIZE = 0x60000000; //  1.5GB

// 时空库后台支持的检索算法类型
enum class AlgorithmType {
    FLAT_COS_INT8,
    FLAT_HAMMING,
    FLAT_IP_FP16,
    FLAT_L2_INT8,
    FLAT_HPP_COS_INT8
};

enum class MemoryStrategy {
    PURE_DEVICE_MEMORY = 0,
    HETERO_MEMORY,
    HPP
};

// 特征属性信息, 入库时和特征向量一起添加
struct FeatureAttr {
    int32_t time;     // 时间戳信息, 调用者确保不超过类型表示的范围
    uint32_t tokenId; // token id, 调用者确保tokenId小于Token总数(tokenNum)
};

// 属性过滤条件, 查询时输入
struct AttrFilter {
    int32_t timesStart;      // 开始时间, 调用者确保不超过类型表示的范围
    int32_t timesEnd;        // 结束时间, 调用者确保不超过类型表示的范围
    uint8_t *tokenBitSet;    // 待查询的token id集合, 按位表示
    uint32_t tokenBitSetLen; // 待查询的token id集合长度, 调用者确保此值不超过tokenNum
};

struct ExtraValAttr {
    int16_t val = 0;             // 附加属性 是否带帽子等
};

struct ExtraValFilter {
    int16_t filterVal = 0;       // 待查询附加属性
    int16_t matchVal = -1;       // 附加属性查询模式 模式0和模式1
};

class AscendIndexTS {
public:
    AscendIndexTS() = default;

    virtual ~AscendIndexTS() = default;

    /* *
     * @brief Init
     * @param[in] deviceId			    Index使用的芯片id
     * @param[in] dim                   底库向量的dim
     * @param[in] tokenNum              Token个数
     * @param[in] algType				底层使用的距离比对算法
     * @param[in] memoryStrategy        底层使用的内存策略
     * @param[in] customAttrLen         自定义属性长度
     * @param[in] customAttrBlockSize   自定义属性blocksize的大小
     * @param[in] maxFeatureRowCount    底库最大向量条数
     * @return							状态码
     */
    APP_ERROR Init(uint32_t deviceId,
                   uint32_t dim,
                   uint32_t tokenNum,
                   AlgorithmType algType = AlgorithmType::FLAT_COS_INT8,
                   MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY,
                   uint32_t customAttrLen = 0,
                   uint32_t customAttrBlockSize = 0,
                   uint64_t maxFeatureRowCount = std::numeric_limits<uint64_t>::max());

    /* *
     * @brief InitWithExtraVal
     * @param[in] deviceId			    Index使用的芯片id
     * @param[in] dim                   底库向量的dim
     * @param[in] tokenNum              Token个数
     * @param[in] resources             共享内存大小
     * @param[in] algType				底层使用的距离比对算法
     * @param[in] memoryStrategy        底层使用的内存策略
     * @param[in] customAttrLen         自定义属性长度
     * @param[in] customAttrBlockSize   自定义属性blocksize的大小
     * @param[in] maxFeatureRowCount    底库最大向量条数
     * @return							状态码
     */
    APP_ERROR InitWithExtraVal(uint32_t deviceId,
                               uint32_t dim,
                               uint32_t tokenNum,
                               uint64_t resources,
                               AlgorithmType algType = AlgorithmType::FLAT_HAMMING,
                               MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY,
                               uint32_t customAttrLen = 0,
                               uint32_t customAttrBlockSize = 0,
                               uint64_t maxFeatureRowCount = std::numeric_limits<uint64_t>::max());

    /* *
     * @brief InitWithQuantify
     * @param[in] deviceId			    Index使用的芯片id
     * @param[in] dim                   底库向量的dim
     * @param[in] tokenNum              Token个数
     * @param[in] resources             共享内存大小
     * @param[in] scale			        缩放因子
     * @param[in] algType				底层使用的距离比对算法
     * @param[in] customAttrLen         自定义属性长度
     * @param[in] customAttrBlockSize   自定义属性blocksize的大小
     * @return							状态码
     */
    APP_ERROR InitWithQuantify(uint32_t deviceId,
                                uint32_t dim,
                                uint32_t tokenNum,
                                uint64_t resources,
                                const float *scale,
                                AlgorithmType algType = AlgorithmType::FLAT_IP_FP16,
                                uint32_t customAttrLen = 0,
                                uint32_t customAttrBlockSize = 0);

    /* *
     * @brief SetHeteroMemParam
     * @param[in] deviceCapacity        异构内存策略下，设备侧存储底库容量（字节）
     * @param[in] deviceBuffer          异构内存策略下，设备侧缓存容量（字节）
     * @param[in] hostCapacity          异构内存策略下，HOST侧存储底库容量（字节）
     * @return                          状态码
     */
    APP_ERROR SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);

    /* *
     * @brief AddFeature       			添加特征
     * @param[in] count                 待添加的特征数量
     * @param[in] features              待添加的特征, 特征向量一律采用uint8_t类型作为输入/输出
     * @param[in] attributes            待添加的特征属性
     * @param[in] labels                待添加的特征label, 使用上需要保证label在index实例中的唯一性
     * @param[in] customAttr            待添加的用户自定义特征属性
     * @return							状态码
     */
    APP_ERROR AddFeature(int64_t count,
                         const void *features,
                         const FeatureAttr *attributes,
                         const int64_t *labels,
                         const uint8_t *customAttr = nullptr);

    /* *
     * @brief AddWithExtraVal       	添加附加属性特征
     * @param[in] count                 待添加的特征数量
     * @param[in] features              待添加的特征, 特征向量一律采用uint8_t类型作为输入/输出
     * @param[in] attributes            待添加的特征属性
     * @param[in] labels                待添加的特征label, 使用上需要保证label在index实例中的唯一性
     * @param[in] extraVal              待添加的附加特征属性
     * @param[in] customAttr            待添加的用户自定义特征属性
     * @return							状态码
     */
    APP_ERROR AddWithExtraVal(int64_t count,
                              const void *features,
                              const FeatureAttr *attributes,
                              const int64_t *labels,
                              const ExtraValAttr *extraVal,
                              const uint8_t *customAttr = nullptr);

    /* *
     * @brief AddFeatureByIndice		按照位置添加特征
     * @param[in] count                 待添加的特征数量
     * @param[in] features              待添加的特征向量
     * @param[in] attributes            待添加的特征属性
     * @param[in] indices               待添加特征在底库中的位置
     * @param[in] customAttr            待添加的用户自定义特征属性
     * @return							状态码
     */
    APP_ERROR AddFeatureByIndice(int64_t count,
                                 const void *features,
                                 const FeatureAttr *attributes,
                                 const int64_t *indices,
                                 const ExtraValAttr *extraVal = nullptr,
                                 const uint8_t *customAttr = nullptr);

    /* *
     * @brief Search		            计算输入特征和底库的距离, 并输出topk个的距离值和下标
     * @param[in] count                 待比较的特征数量
     * @param[in] features              待比较的特征
     * @param[in] attrFilter            属性过滤信息, 当shareAttrFilter为True时, 个数为1, 当shareAttrFilter为False时, 个数为count
     * @param[in] shareAttrFilter       待比较的特征向量是否共享属性过滤规则
     * @param[in] topk                  每个待比较特征查询后期望输出的topk值
     * @param[out] labels			    查询后输出的底库特征向量对应的label值, 由调用者分配内存, 大小为topk*count个
     * @param[out] distances            查询后输出的底库特征向量对应的distance值, 由调用者分配内存, 大小为topk*count个
     * @param[out] validNums            每个待比较特征向量实际输出的有效个数, 小于等于topk, 由调用者分配内存, 大小为count个
     * @param[in] enableTimeFilter      是否进行时间属性过滤
     * @return						    状态码
     */
    APP_ERROR Search(uint32_t count,
                     const void *features,
                     const AttrFilter *attrFilter,
                     bool shareAttrFilter,
                     uint32_t topk,
                     int64_t *labels,
                     float *distances,
                     uint32_t *validNums,
                     bool enableTimeFilter = true);

    /* *
     * @brief SearchWithExtraVal		计算输入特征和底库的距离, 并输出topk个的距离值和下标
     * @param[in] count                 待比较的特征数量
     * @param[in] features              待比较的特征
     * @param[in] attrFilter            属性过滤信息, 当shareAttrFilter为True时, 个数为1, 当shareAttrFilter为False时, 个数为count
     * @param[in] shareAttrFilter       待比较的特征向量是否共享属性过滤规则
     * @param[in] topk                  每个待比较特征查询后期望输出的topk值
     * @param[out] labels			    查询后输出的底库特征向量对应的label值, 由调用者分配内存, 大小为topk*count个
     * @param[out] distances            查询后输出的底库特征向量对应的distance值, 由调用者分配内存, 大小为topk*count个
     * @param[out] validNums            每个待比较特征向量实际输出的有效个数, 小于等于topk, 由调用者分配内存, 大小为count个
     * @param[in] extraValFilter        附加属性过滤信息
     * @param[in] enableTimeFilter      是否进行时间属性过滤
     * @return						    状态码
     */
    APP_ERROR SearchWithExtraVal(uint32_t count,
                                 const void *features,
                                 const AttrFilter *attrFilter,
                                 bool shareAttrFilter,
                                 uint32_t topk,
                                 int64_t *labels,
                                 float *distances,
                                 uint32_t *validNums,
                                 const ExtraValFilter *extraValFilter,
                                 bool enableTimeFilter = true);

    /* *
     * @brief SearchWithExtraMask       计算输入特征和底库的距离, 支持用户输入外部的过滤Mask, 并输出topk个的距离值和下标
     * @param[in] count                 待比较的特征数量
     * @param[in] features              待比较的特征
     * @param[in] attrFilter            属性过滤信息, 当shareAttrFilter为True时, 个数为1, 当shareAttrFilter为False时, 个数为count
     * @param[in] shareAttrFilter       待比较的特征向量是否共享属性过滤规则
     * @param[in] topk                  每个待比较特征查询后期望输出的topk值
     * @param[in] extraMask             用户自定义逻辑生成的距离值过滤Mask, 由调用者分配内存, 长度为count*extraMaskLenEachQuery个字节,
                                        当shareAttrFilter为True时, 长度为1*extraMaskLenEachQuery个字节
     * @param[in] extraMaskLenEachQuery 每个待比较特征向量对应的自定义过滤Mask长度, 单位为字节
     * @param[in] extraMaskIsAtDevice   用户自定义extraMask的数据是否属于device侧内存
     * @param[out] labels			    查询后输出的底库特征向量对应的label值, 由调用者分配内存, 大小为topk*count个
     * @param[out] distances            查询后输出的底库特征向量对应的distance值, 由调用者分配内存, 大小为topk*count个
     * @param[out] validNums            每个待比较特征向量实际输出的有效个数, 小于等于topk, 由调用者分配内存, 大小为count个
     * @return						    状态码
     */
    APP_ERROR SearchWithExtraMask(uint32_t count,
                                  const void *features,
                                  const AttrFilter *attrFilter,
                                  bool shareAttrFilter,
                                  uint32_t topk,
                                  const uint8_t *extraMask,
                                  uint64_t extraMaskLenEachQuery,
                                  bool extraMaskIsAtDevice,
                                  int64_t *labels,
                                  float *distances,
                                  uint32_t *validNums,
                                  bool enableTimeFilter = true);

    /* *
     * @brief SearchWithExtraMask  计算输入特征和底库的距离, 支持用户输入外部的过滤Mask和额外相似度，并输出topk个的距离值和下标
     * @param[in] count                         待比较的特征数量
     * @param[in] features                      待比较的特征
     * @param[in] attrFilter                    属性过滤信息, 当shareAttrFilter为True时, 个数为1, 当shareAttrFilter为False时, 个数为count
     * @param[in] shareAttrFilter               待比较的特征向量是否共享属性过滤规则
     * @param[in] topk                          每个待比较特征查询后期望输出的topk值
     * @param[in] extraMask                     用户自定义逻辑生成的距离值过滤Mask, 由调用者分配内存, 长度为count*extraMaskLenEachQuery个字节,
                                                当shareAttrFilter为True时, 长度为1*extraMaskLenEachQuery个字节
     * @param[in] extraMaskLenEachQuery         每个待比较特征向量对应的自定义过滤Mask长度, 单位为字节
     * @param[in] extraMaskIsAtDevice           用户自定义extraMask的数据是否属于device侧内存
     * @param[in] extraScore                    用户输入的额外相似度，长度为count*totalPad（totalPad为底库大小按照16对齐的值）
     * @param[out] labels			            查询后输出的底库特征向量对应的label值, 由调用者分配内存, 大小为topk*count个
     * @param[out] distances                    查询后输出的底库特征向量对应的distance值, 由调用者分配内存, 大小为topk*count个
     * @param[out] validNums                    每个待比较特征向量实际输出的有效个数, 小于等于topk, 由调用者分配内存, 大小为count个
     * @return						            状态码
     */
    APP_ERROR SearchWithExtraMask(uint32_t count,
                                  const void *features,
                                  const AttrFilter *attrFilter,
                                  bool shareAttrFilter,
                                  uint32_t topk,
                                  const uint8_t *extraMask,
                                  uint64_t extraMaskLenEachQuery,
                                  bool extraMaskIsAtDevice,
                                  const uint16_t *extraScore,
                                  int64_t *labels,
                                  float *distances,
                                  uint32_t *validNums,
                                  bool enableTimeFilter = true);

    /* *
     * @brief GetFeatureNum			获取该Index实例中特征向量的个数
     * @param[out] totalNum			特征向量的条数
     * @return						状态码
     */
    APP_ERROR GetFeatureNum(int64_t *totalNum) const;

    /* *
     * @brief GetBaseByRange		获取该Index实例中所有特征及FeatureAttr特征属性
     * @param[in] offset			获取底库特征初始偏移值
     * @param[in] num			    特征数量
     * @param[out] labels			特征label
     * @param[out] features			特征
     * @param[out] attributes		特征属性
     * @return						状态码
     */
    APP_ERROR GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes);

    /* *
     * @brief GetBaseByRangeWithExtraVal	获取该Index实例中所有特征及特征属性
     * @param[in] offset			        获取底库特征初始偏移值
     * @param[in] num			            特征数量
     * @param[out] labels			        特征label
     * @param[out] features			        特征
     * @param[out] attributes		        特征属性
     * @param[out] extraVal		            附加属性
     * @return						        状态码
     */
    APP_ERROR GetBaseByRangeWithExtraVal(uint32_t offset,
                                         uint32_t num,
                                         int64_t *labels,
                                         void *features,
                                         FeatureAttr *attributes,
                                         ExtraValAttr *extraVal) const;
    /* *
     * @brief GetFeatureByLabel			获取指定label的特征
     * @param[in] count					获取特征的数量
     * @param[in] labels				特征label
     * @param[out] features             特征
     * @return							状态码
     */
    APP_ERROR GetFeatureByLabel(int64_t count, const int64_t *labels, void *features) const;

    /* *
     * @brief GetFeatureAttrByLabel		获取指定label的特征属性
     * @param[in] count					获取特征的数量
     * @param[in] labels				特征label
     * @param[out] attributes           特征属性
     * @return							状态码
     */
    APP_ERROR GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const;

    /* *
     * @brief GetExtraValAttrByLabel	获取指定label的附加属性
     * @param[in] count					获取特征的数量
     * @param[in] labels				特征label
     * @param[out] extraVal             附加属性
     * @return							状态码
     */
    APP_ERROR GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const;

    /* *
     * @brief GetFeatureByIndice	    获取指定indice的特征和属性信息
     * @param[in] count					获取特征的数量
     * @param[in] indices				特征所在的位置
     * @param[out] labels				特征label
     * @param[out] features             特征，为nullptr则不获取，否则需要预先申请合理空间
     * @param[out] attributes           特征属性，为nullptr则不获取，否则需要预先申请合理空间
     * @param[out] extraVal             附加属性，为nullptr则不获取，否则需要预先申请合理空间
     * @return							状态码
     */
    APP_ERROR GetFeatureByIndice(int64_t count,
                                 const int64_t *indices,
                                 int64_t *labels = nullptr,
                                 void *features = nullptr,
                                 FeatureAttr *attributes = nullptr,
                                 ExtraValAttr *extraVal = nullptr) const;

    /* *
     * @brief FastDeleteFeatureByIndice  标记删除指定indices位置的特征
     * @param[in] count					 待删除的特征数量
     * @param[in] indices				 待删除的特征所在的位置
     * @return						 	 状态码
     */
    APP_ERROR FastDeleteFeatureByIndice(int64_t count, const int64_t *indices);

    /* *
     * @brief FastDeleteFeatureByRange   标记删除指定连续位置的特征
     * @param[in] start					 待删除的特征的起始位置
     * @param[in] count				     待删除的特征数量
     * @return						 	 状态码
     */
    APP_ERROR FastDeleteFeatureByRange(int64_t start, int64_t count);

    /* *
     * @brief GetBaseMask		获取底库对应的掩码
     * @param[in] count			mask数组长度，先获取
     * @param[out] mask			底库的掩码，对应bit位为1表示特征有效，为0表示特征无效
     * @return					状态码
     */
    APP_ERROR GetBaseMask(int64_t count, uint8_t *mask) const;

    /* *
     * @brief GetCustomAttrByBlockId		获取指定blockId的自定义属性
     * @param[in] blockId					待获取的blockId
     * @param[out] customAttr               用户自定义特征属性
     * @return							    状态码
     */
    APP_ERROR GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const;

    /* *
     * @brief DeleteFeatureByLabel		批量移除指定label的特征
     * @param[in] count					待移除的特征数量
     * @param[in] labels				特征label
     * @return							状态码
     */
    APP_ERROR DeleteFeatureByLabel(int64_t count, const int64_t *labels);

    /* *
     * @brief DeleteFeatureByToken		批量移除指定Toeken ID对应的特征
     * @param[in] count					待移除的token数量
     * @param[in] tokens  				Token ID号
     * @return							状态码
     */
    APP_ERROR DeleteFeatureByToken(int64_t count, const uint32_t *tokens);

    APP_ERROR SetSaveHostMemory();

    AscendIndexTS(const AscendIndexTS &) = delete;
    AscendIndexTS &operator=(const AscendIndexTS &) = delete;

private:
    std::shared_ptr<::ascend::TSSuperBase> pImpl = nullptr;
    uint32_t deviceId = 0;
    uint32_t maxTokenNum = 0;
    mutable std::mutex mtx;
    MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY;
    bool heteroParamSetFlag = false;
};
} // namespace ascend
} // namespace faiss
#endif