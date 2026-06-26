# Approximate Retrieval<a name="ZH-CN_TOPIC_0000001482524834"></a>

## `AscendIndexBinaryFlat`<a name="ZH-CN_TOPIC_0000001506334701"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456694988"></a>

The `AscendIndexBinaryFlat` class inherits from Faiss `IndexBinary` and is used for binary feature retrieval.

It supports only Atlas Inference Series products.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `add`<a name="ZH-CN_TOPIC_0000001456854896"></a>

| API Definition | `void add(idx_t n, const uint8_t *x) override;` |
| --- | --- |
| Description | Adds feature vectors to the base library. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const uint8_t *x`: Feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims/8 * n`. Otherwise, out-of-bounds reads or writes may occur or the program may crash. `n > 0`. The `add` operation must ensure that the final base library size `ntotal` is the smaller of the actual chip memory capacity and `1e9`. |

> [!NOTE]
>
>- The `add` API cannot be used together with the `add_with_ids` API.
>- After you use the `add` API, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` API.

### `add_with_ids`<a name="ZH-CN_TOPIC_0000001506414809"></a>

| API Definition | `void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;` |
| --- | --- |
| Description | Adds feature vectors to the base library and specifies the corresponding IDs. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const uint8_t *x`: Feature vectors to add to the base library.<br>`const idx_t *xids`: IDs of the feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | `0 < n`. The `add` operation must ensure that the final base library size `n` is the smaller of the actual chip memory capacity and `1e9`. The length of pointer `x` must be `dims/8 * n`, and the length of pointer `xids` must be `n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that `xids` is valid according to your service scenario. If duplicate IDs exist in the base library, the labels in the search results cannot be mapped to specific base-library vectors. |

### `AscendIndexBinaryFlat`<a name="ZH-CN_TOPIC_0000001456535056"></a>

| API Definition | `AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);` |
| --- | --- |
| Description | Constructor of `AscendIndexBinaryFlat`. It creates an `AscendIndexBinaryFlat` with dimension `dims` and sets device-side resources based on the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndexBinaryFlat`.<br>`AscendIndexBinaryFlatConfig config`: Device-side resource configuration.<br>`bool usedFloat`: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the `search` API. The default value is `false`. Set it to `true` to enable the performance improvement. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ { 256, 512, 1024 }. |

<a name="table191641015539"></a>

| API Definition | `AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);` |
| --- | --- |
| Description | Constructor of `AscendIndexBinaryFlat`. It creates an Ascend retrieval index based on an existing `index`. |
| Input | `const faiss::IndexBinaryFlat *index`: CPU-side index resource.<br>`AscendIndexBinaryFlatConfig config`: Device-side resource configuration.<br>`bool usedFloat`: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the `search` API. The default value is `false`. Set it to `true` to enable the performance improvement. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU index pointer. `index->d` ∈ {256, 512, 1024}. `index->ntotal` is the smaller of the actual chip memory capacity and `1e9`. |

<a name="table142022518319"></a>

| API Definition | `AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);` |
| --- | --- |
| Description | Constructor of `AscendIndexBinaryFlat`. It creates an Ascend retrieval index based on an existing `index`. |
| Input | `const faiss::IndexBinaryIDMap *index`: CPU-side index resource.<br>`AscendIndexBinaryFlatConfig config`: Device-side resource configuration.<br>`bool usedFloat`: Used to improve performance for the retrieval mode in which binary features are stored and float features are used for retrieval, that is, the `search` API. The default value is `false`. Set it to `true` to enable the performance improvement. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `faiss::IndexBinaryIDMap` pointer. `index->index` must be a valid `IndexBinaryFlat` pointer. `index->index->d` ∈ {256, 512, 1024}. `index->index->ntotal` is the smaller of the actual chip memory capacity and `1e9`. |

<a name="table145324411437"></a>

| API Definition | `AscendIndexBinaryFlat(const AscendIndexBinaryFlat &) = delete;` |
| --- | --- |
| Description | Declares the copy constructor of `AscendIndexBinaryFlat` as deleted. Therefore, `AscendIndexBinaryFlat` is a non-copyable type. |
| Input | `const AscendIndexBinaryFlat &`: Constant `AscendIndexBinaryFlat`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexBinaryFlat`<a name="ZH-CN_TOPIC_0000001506495917"></a>

<a name="table13115573310"></a>

| API Definition | `virtual ~AscendIndexBinaryFlat() = default;` |
| --- | --- |
| Description | Destructor of `AscendIndexBinaryFlat`. It destroys the `AscendIndexBinaryFlat` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001506414941"></a>

| API Definition | `void copyFrom(const faiss::IndexBinaryFlat *index);` |
| --- | --- |
| Description | Copies data from an existing `Index` to `AscendIndexBinaryFlat`, clears the current base library of `AscendIndexBinaryFlat`, and retains the original device-side resource configuration. |
| Input | `const faiss::IndexBinaryFlat *index`: `faiss::IndexBinaryFlat` pointer. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexBinaryFlat` pointer. `index->d` ∈ {256, 512, 1024}. `index->ntotal` is the smaller of the actual chip memory capacity and `1e9`. |

<a name="table1570816514419"></a>

| API Definition | `void copyFrom(const faiss::IndexBinaryIDMap *index);` |
| --- | --- |
| Description | Copies data from an existing `index` to `AscendIndexBinaryFlat`, clears the current base library of `AscendIndexBinaryFlat`, and retains the original device-side resource configuration. |
| Input | `const faiss::IndexBinaryIDMap *index`: `faiss::IndexBinaryIDMap` pointer. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `faiss::IndexBinaryIDMap` pointer. `index->index` must be a valid `IndexBinaryFlat` pointer. `index->index->d` ∈ {256, 512, 1024}. `index->index->ntotal` is the smaller of the actual chip memory capacity and `1e9`. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001456855048"></a>

| API Definition | `void copyTo(faiss::IndexBinaryFlat *index) const;` |
| --- | --- |
| Description | Copies data from an existing `AscendIndexBinaryFlat` to `faiss::IndexBinaryFlat index`, and clears the original resources of `index`. |
| Input | `faiss::IndexBinaryFlat *index`: `faiss::IndexBinaryFlat` pointer. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexBinaryFlat` pointer. The user must release the resources of the copied `index`. |

<a name="table19831553111512"></a>

| API Definition | `void copyTo(faiss::IndexBinaryIDMap *index) const;` |
| --- | --- |
| Description | Copies data from an existing `AscendIndexBinaryFlat` to `faiss::IndexBinaryIDMap index`, and clears the original resources of `index`. |
| Input | `faiss::IndexBinaryIDMap *index`: `faiss::IndexBinaryIDMap` pointer. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexBinaryIDMap` pointer. The user must release the copied `Index` resources. |

### `operator=`<a name="ZH-CN_TOPIC_0000001456535072"></a>

| API Definition | `AscendIndexBinaryFlat &operator = (const AscendIndexBinaryFlat &) = delete;` |
| --- | --- |
| Description | Declares the assignment constructor of `AscendIndexBinaryFlat` as deleted. Therefore, `AscendIndexBinaryFlat` is a non-copyable type. |
| Input | `const AscendIndexBinaryFlat &`: Constant `AscendIndexBinaryFlat`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `remove_ids`<a name="ZH-CN_TOPIC_0000001506495769"></a>

| API Definition | `size_t remove_ids(const faiss::IDSelector &sel) override;` |
| --- | --- |
| Description | Deletes the specified feature vectors from the base library. |
| Input | `const faiss::IDSelector &sel`: Feature vectors to delete. For details about usage and definition, see the relevant Faiss documentation. |
| Output | None |
| Returns | Number of feature vectors deleted successfully, with invalid IDs ignored. |
| Constraints | None |

### `reset`<a name="ZH-CN_TOPIC_0000001456855028"></a>

| API Definition | `void reset() override;` |
| --- | --- |
| Description | Clears the base-library vectors of this `AscendIndexBinaryFlat`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `search`<a id="ZH-CN_TOPIC_0000001456375288"></a>

| API Definition | `void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels, const SearchParameters *params) const override;` |
| --- | --- |
| Description | Feature vector query API. It returns the IDs and corresponding distances of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n`: Number of query vectors.<br>`const uint8_t *x`: Query vectors.<br>`idx_t k`: Number of most similar results to return.<br>`const SearchParameters *params`: Optional Faiss parameters. The default value is `nullptr`, and this parameter is not supported for now. |
| Output | `int32_t *distances`: Distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: IDs of the `k` nearest vectors. |
| Returns | None |
| Constraints | The length of feature vector data `x` must be `dims/8 * n`, and the lengths of `distances` and `labels` must be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `0 < n ≤ 1e9`, `0 < k ≤ 1e5`. The `n ≤ 1e9` limit is far beyond the actual available resources, so you are advised to choose an appropriate number of query vectors according to your service scenario. |

<a name="table1659211341612"></a>

| API Definition | `void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;` |
| --- | --- |
| Description | Feature vector query API. It returns the IDs and corresponding distances of the `k` most similar features based on the input feature vectors. This API is used for the retrieval mode in which binary features are stored in the base library and float features are used for retrieval. |
| Input | `idx_t n`: Number of query vectors.<br>`const float *x`: Query vectors.<br>`idx_t k`: Number of most similar results to return. |
| Output | `float *distances`: Distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: IDs of the `k` nearest vectors. |
| Returns | None |
| Constraints | The length of feature vector data `x` must be `dims * n`, and the lengths of `distances` and `labels` must be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `0 < n ≤ 1e9`, `0 < k ≤ 1e5`. The `n ≤ 1e9` limit is far beyond the actual available resources, so you are advised to choose an appropriate number of query vectors according to your service scenario. |

### `setRemoveFast`<a name="ZH-CN_TOPIC_0000002024780673"></a>

| API Definition | `static void setRemoveFast(bool removeFast);` |
| --- | --- |
| Description | Sets whether to quickly delete vectors from the base library. |
| Input | `bool removeFast`: Set it to `true` to use fast deletion, or `false` not to use it. |
| Output | None |
| Returns | None |
| Constraints | Fast deletion improves the performance of deleting the base library, but it slightly reduces the performance of adding data to the base library. If you do not call this API, fast deletion is disabled by default. This API can be called only once, and you must call it before you construct the index object. |

## `AscendIndexBinaryFlatConfig`<a name="ZH-CN_TOPIC_0000001506495777"></a>

`AscendIndexBinaryFlat` uses the corresponding `AscendIndexBinaryFlatConfig` to initialize the corresponding resources and configure the device-side hardware resources `devices` and the preset memory pool size `resources` during retrieval.

- `AscendIndexBinaryFlat` supports only Atlas Inference Series products with a single Ascend AI Processor. It depends on the AICPU operator and the BinaryFlat operator. See [Introduction to Custom Operators](../user_guide.md#generating-operators) to generate the corresponding operators.
- `AscendIndexBinaryFlat` supports only standard deployment mode.

**Members<a name="section1372191465013"></a>**

|Member|Type|Description|
|--|--|--|
|deviceList|std::vector\<int>|Device-side device IDs. The `AscendIndexBinaryFlat` class supports only a single accelerator card of the Atlas Inference Series products.|
|resourceSize|int64_t|Size of the device-side memory pool, in bytes. The default value is 1024 MB. The valid range is [1024*1024*1024, 32*1024*1024*1024]. For a base library with 10 million vectors, 5 GB is recommended.|

**API Description<a name="section108610580175"></a>**

| API Definition | `AscendIndexBinaryFlatConfig() = default;` |
| --- | --- |
| Description | Default constructor. The default value of `devices` is `{ 0 }`, which uses the 0th Ascend AI Processor for computation. The default value of `resources` is 1024 MB. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | `AscendIndexBinaryFlat` supports only Atlas Inference Series products with a single Ascend AI Processor. If the 0th Ascend AI Processor is unavailable, you cannot use the default constructor. |

<a name="table092314378186"></a>

| API Definition | `AscendIndexBinaryFlatConfig(std::initializer_list<int> devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor that uses `initializer_list` for `devices`. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs. For this class, only a single device is supported, that is, the length of `devices` must be 1.<br>`int64_t resources`: Preset memory pool size. The default value is 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, non-duplicated device IDs, and the length must be 1. The valid range of `resources` is [1024*1024*1024, 32*1024*1024*1024]. For a 10 million base library, 5 GB is recommended. |

<a name="table1743710521181"></a>

| API Definition | `AscendIndexBinaryFlatConfig(std::vector<int> devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor that uses `vector` for `devices`. |
| Input | `std::vector<int> devices`: Device-side device IDs. For this class, only a single device is supported, that is, the length of `devices` must be 1.<br>`int64_t resources`: Preset memory pool size. The default value is 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, non-duplicated device IDs, and the length must be 1. The valid range of `resources` is [1024*1024*1024, 32*1024*1024*1024]. For a 10 million base library, 5 GB is recommended. |

## `AscendIndexIVF`<a name="ZH-CN_TOPIC_0000001456375220"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506334721"></a>

`AscendIndexIVF` serves as the base class of IVF-based indexes in the feature retrieval component and defines APIs for other IVF indexes in feature retrieval.

For IVF algorithms, the linear scaling on the Atlas 300I Duo inference card depends on the proportion of distance-computation workload in the entire search process. Compared with other computation types, only the distance-computation workload can be evenly distributed across multiple compute units. Therefore, scaling is better in large-batch and large-`nprobe` scenarios, and worse in small-batch and small-`nprobe` scenarios.

> [!NOTE]
> IVF algorithms should follow the rule `nlist * 2MB + resourceSize < NPU-side memory` to avoid memory allocation failures at runtime. For example, if the memory on the NPU card is 64 GB, `nlist` should be smaller than 32768. Since `32768 * 2MB = 64GB`, runtime may exceed the NPU memory size. This limit exists because the current retrieval service prioritizes large-page memory, and the allocation granularity of large-page memory is 2 MB. When every bucket in `nlist` contains data, the hardware allocates memory aligned to the 2 MB granularity. `resourceSize` is the shared memory size specified by the user in `AscendIndexIVFConfig`, and the default value is 128 MB.

### `AscendIndexIVF`<a name="ZH-CN_TOPIC_0000001506414821"></a>

| API Definition | `AscendIndexIVF(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFConfig config = AscendIndexIVFConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexIVF`. It creates `AscendIndexIVF` and sets device-side resources based on the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndexIVF`.<br>`faiss::MetricType metric`: Distance metric used by `AscendIndex` when performing feature-vector similarity retrieval. The current supported values are `faiss::MetricType::METRIC_L2` and `faiss::MetricType::METRIC_INNER_PRODUCT`.<br>`int nlist`: Number of clustering centers. This corresponds to the `coarse_centroid_num` parameter in the operator generation script.<br>`AscendIndexIVFConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `nlist` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. |

<a name="table9624174810199"></a>

| API Definition | `AscendIndexIVF(const AscendIndexIVF&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor of this index as deleted. Therefore, it is a non-copyable type. |
| Input | `const AscendIndexIVF&`: Constant `AscendIndexIVF`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexIVF`<a name="ZH-CN_TOPIC_0000001506334765"></a>

| API Definition | `virtual ~AscendIndexIVF();` |
| --- | --- |
| Description | Destructor of `AscendIndexIVF`. It destroys the `AscendIndexIVF` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001506334601"></a>

| API Definition | `void copyFrom(const faiss::IndexIVF* index);` |
| --- | --- |
| Description | `AscendIndexIVF` copies data from an existing `index` to Ascend and retains the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexIVF* index`: CPU-side index resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The `probe` value of this `index` must be greater than 0 and less than or equal to `nlist`. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001506615113"></a>

| API Definition | `void copyTo(faiss::IndexIVF* index) const;`|
| --- | --- |
| Description | Copies the retrieval resources of `AscendIndexIVF` to the CPU side. |
| Input | `faiss::IndexIVF* index`: CPU-side index resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The resources occupied by `Index` are released by the user. |

### `getNumLists`<a name="ZH-CN_TOPIC_0000001506614893"></a>

| API Definition | `int getNumLists() const;` |
| --- | --- |
| Description | Returns the current `nlist` value of `AscendIndexIVF`. |
| Input | None |
| Output | None |
| Returns | `nlist` value of `AscendIndexIVF`. |
| Constraints | None |

### `getNumProbes`<a name="ZH-CN_TOPIC_0000001456534948"></a>

| API Definition | `int getNumProbes() const;` |
| --- | --- |
| Description | Returns the current `nprobe` value of `AscendIndexIVF`. |
| Input | None |
| Output | None |
| Returns | `nprobe` value of `AscendIndexIVF`. |
| Constraints | None |

### `getListCodesAndIds`<a name="ZH-CN_TOPIC_0000001456854940"></a>

| API Definition | `virtual void getListCodesAndIds(int listId, std::vector<uint8_t>& codes, std::vector<ascend_idx_t>& ids) const;` |
| --- | --- |
| Description | Returns the feature vectors and corresponding IDs at a specific `nlistId` in the current `AscendIndexIVF`. |
| Input | `int listId`: Specific `nlistId` in the `nlist` of `AscendIndexIVF`. |
| Output | `std::vector<uint8_t>& codes`: Feature vectors at the specific `nlistId` in the `nlist` of `AscendIndexIVF`.<br>`std::vector<ascend_idx_t>& ids`: IDs of the feature vectors at the specific `nlistId` in the `nlist` of `AscendIndexIVF`. |
| Returns | None |
| Constraints | `0 ≤ listId < nlist`. |

### `getListLength`<a name="ZH-CN_TOPIC_0000001506614973"></a>

| API Definition | `virtual uint32_t getListLength(int listId) const;` |
| --- | --- |
| Description | Returns the length of a specific `nlistId` in the current `nlist` of `AscendIndexIVF`. |
| Input | `int listId`: Specific `nlistId` in the `nlist` of `AscendIndexIVF`. |
| Output | None |
| Returns | Length of the specific `nlistId` in the `nlist` of `AscendIndexIVF`. |
| Constraints | `0 ≤ listId < nlist`. |

### `operator=`<a name="ZH-CN_TOPIC_0000001506495837"></a>

| API Definition | `AscendIndexIVF& operator=(const AscendIndexIVF&) = delete;` |
| --- | --- |
| Description | Declares the assignment constructor of this index as deleted. Therefore, it is a non-copyable type. |
| Input | `const AscendIndexIVF&`: Constant `AscendIndexIVF`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `reclaimMemory`<a name="ZH-CN_TOPIC_0000001506615049"></a>

| API Definition | `size_t reclaimMemory() override;` |
| --- | --- |
| Description | Reduces the memory occupied by the base library without changing the number of base-library entries. This API inherits from `AscendIndex` and provides a concrete implementation. |
| Input | None |
| Output | None |
| Returns | Amount of memory reduced, in bytes. |
| Constraints | None |

### `reserveMemory`<a name="ZH-CN_TOPIC_0000001506334617"></a>

| API Definition | `void reserveMemory(size_t numVecs) override;` |
| --- | --- |
| Description | Abstract API that reserves memory for the base library before the base library is built. This API inherits from `AscendIndex` and provides a concrete implementation. |
| Input | `size_t numVecs`: Number of base-library vectors for which to reserve memory. |
| Output | None |
| Returns | None |
| Constraints | In a single-card environment: `0 < numVecs ≤ 2e8`. In a multi-card environment: `0 < numVecs ≤ 1e9` (`numVecs` divided by the number of cards must be smaller than `2e8`). Exceeding the limit throws an exception and stops the program. |

### `reset`<a name="ZH-CN_TOPIC_0000001506414685"></a>

| API Definition | `void reset() override;` |
| --- | --- |
| Description | Clears the base-library vectors of this `AscendIndexIVF`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `setNumProbes`<a name="ZH-CN_TOPIC_0000001506614937"></a>

| API Definition | `virtual void setNumProbes(int nprobes);` |
| --- | --- |
| Description | Sets the current `nprobe` value of `AscendIndexIVF`. |
| Input | `int nprobes`: `nprobe` value of `AscendIndexIVF`. |
| Output | None |
| Returns | None |
| Constraints | `0 < nprobes ≤ nlist`. |

## `AscendIndexIVFConfig`<a name="ZH-CN_TOPIC_0000001456535024"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456695128"></a>

`AscendIndexIVF` uses the corresponding `AscendIndexIVFConfig` to initialize the corresponding resources.

**Members<a name="section1372191465013"></a>**

|Member|Type|Description|
|--|--|--|
|flatConfig|AscendIndexConfig|Parameter configuration object.|
|useKmeansPP|bool|Whether to use NPU acceleration for the IVF clustering process.|
|cp|ClusteringParameters|Clustering-related parameters. For details, see the relevant Faiss API documentation. You are not advised to modify this parameter. The default number of training iterations is 16. Setting the number of iterations too large significantly increases the training time.|

> [!NOTE]
>
> `AscendIndexIVFConfig` inherits from [AscendIndexConfig](./full_retrieval.md#ascendindexconfig)

### `AscendIndexIVFConfig`<a name="ZH-CN_TOPIC_0000001506334629"></a>

<a name="table1319620316150"></a>

| API Definition | `inline AscendIndexIVFConfig();` |
| --- | --- |
| Description | Default constructor. The default value of `devices` is `{0}`, which uses the 0th Ascend AI Processor for computation. The default value of `resources` is 128 MB. The default value of `useKmeansPP` is `false`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table3725347611"></a>

| API Definition | `inline AscendIndexIVFConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor of `AscendIndexIVFConfig`. It creates `AscendIndexIVFConfig`, sets the device-side Ascend AI Processor resources according to the values configured in `devices`, configures the memory pool size, and sets the default number of iterations. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: Preset memory pool size on the device side, in bytes. This memory space stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default parameter is `IVF_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base library size and the search batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, non-duplicated device IDs, and the maximum number is 64. The configured `resourceSize` cannot exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). When it is set to `-1`, the device-side Ascend AI Processor resource configuration uses the default value of 128 MB. |

<a name="table745471811619"></a>

| API Definition | `inline AscendIndexIVFConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor of `AscendIndexIVFConfig`. It creates `AscendIndexIVFConfig`, sets the device-side Ascend AI Processor resources according to the values configured in `devices`, configures the memory pool size, and sets the default number of iterations. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: Preset memory pool size on the device side, in bytes. This memory space stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default parameter is `IVF_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base library size and the search batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, non-duplicated device IDs, and the maximum number is 64. The configured `resourceSize` cannot exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). When it is set to `-1`, the device-side Ascend AI Processor resource configuration uses the default value of 128 MB. |

### `SetDefaultClusteringConfig`<a name="ZH-CN_TOPIC_0000001506495669"></a>

| API Definition | `inline void SetDefaultClusteringConfig();` |
| --- | --- |
| Description | Sets the number of iterations for `AscendIndexIVF` to the default value 10. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendIndexIVFSP`<a name="ZH-CN_TOPIC_0000001635576081"></a>

### Overview<a name="ZH-CN_TOPIC_0000001635815481"></a>

The Ascend-native IVFSP retrieval algorithm uses an in-house matrix approximation strategy to compress feature vectors before storing them in the base library. It then uses an in-house inverted-list strategy to select the base-library entries most likely to contain the ground truth. Finally, it uses an in-house retrieval strategy on the filtered base library to obtain the top K vector results.

`AscendIndexIVFSP` supports only standard mode scenarios and Atlas Inference Series products.

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `add`<a name="ZH-CN_TOPIC_0000001585895568"></a>

| API Definition | `void add(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Adds feature vectors to the base library. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const float *x`: Feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The total number of base-library vectors, `n`, is usually greater than 0 and less than `1e9`. The amount of data added at one time must be smaller than or equal to the base-library data size. |

> [!NOTE]
>
>- The `add` API cannot be used together with the `add_with_ids` API.
>- After you use the `add` API, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` API.
>- The `add` API is optimized for small-batch addition scenarios. In this scenario, accuracy may decrease depending on the dataset. You are advised to use small-batch addition when a base library already exists.

### `add_with_ids`<a name="ZH-CN_TOPIC_0000001586055512"></a>

| API Definition | `void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;` |
| --- | --- |
| Description | Adds feature vectors to the base library and specifies the corresponding IDs. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const float *x`: Feature vectors to add to the base library.<br>`const idx_t *ids`: IDs of the feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`, and the length of pointer `ids` must be `n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. You need to ensure that `ids` is valid according to your service scenario. If duplicate IDs exist in the base library, the `label` in the retrieval results cannot be mapped to a specific base-library vector.<br>The value range of `n` is `0 < n < 1e9`. |

> [!NOTE]
> The `add_with_ids` API is optimized for small-batch addition scenarios. In this scenario, accuracy may decrease depending on the dataset. You are advised to use small-batch addition when a base library already exists.

### `AscendIndexIVFSP`<a name="ZH-CN_TOPIC_0000001585736168"></a>

> [!NOTE]
>
> - Before you pass parameter `config` to the function, set the values of `conf.handleBatch`, `conf.nprobe`, and `conf.searchListSize` according to the actual situation. For field descriptions, see [Common Parameters](#ZH-CN_TOPIC_0000001635696057).
> - The values of `conf.handleBatch` and `conf.searchListSize` must be consistent with the `nprobe handle batch` and `search list size` values used when generating the [IVFSP](../user_guide.md#ivfsp) service operator model file.
> - `conf.filterable`, inherited from [AscendIndexConfig](./full_retrieval.md#ascendindexconfig) false by default. If you want to use the `search_with_filter()` API, set `conf.filterable = true`. Setting `conf.filterable` to `true` stores extra information on the NPU card and consumes more NPU-side memory.

| API Definition | `AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexIVFSP`. It sets device-side resources based on the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndexIVFSP`.<br>`int nonzeroNum`: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.<br>`int nlist`: Number of clustering centers. This corresponds to the value of the `<centroid num>` parameter in the generation of the IVFSP service operator model file.<br>`const char *codeBookPath`: Path of the codebook file used by IVFSP.<br>`faiss::ScalarQuantizer::QuantizerType qType`: Scalar quantization type. The current supported value is only `ScalarQuantizer::QuantizerType::QT_8bit`.<br>`faiss::MetricType metric`: Distance metric used by `AscendIndex` when performing feature-vector similarity retrieval. The current `faiss::MetricType metric` supports only `METRIC_L2`.<br>`AscendIndexIVFSPConfig`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | The values of `<dim>`, `<nonzero num>`, and `<centroid num>` used when training and generating the codebook must correspond to the `dims`, `nonzeroNum`, and `nlist` parameters of this function. The codebook loaded from `codeBookPath` must correspond to the `dims`, `nonzeroNum`, and `nlist` parameters of this function, and the user who runs the program must be the owner of the codebook file. The codebook file cannot be a symbolic link. When `dims` ∈ {64, 128, 256}, `nlist` ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When `dims` ∈ {512, 768}, `nlist` ∈ {256, 512, 1024, 2048}. `nonzeroNum` must be a multiple of 16 and less than or equal to `min(128, dims)`. `metric` ∈ {`faiss::MetricType::METRIC_L2`}. |

<a name="table49022324218"></a>

| API Definition | `AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &codeBookSharedIdx, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexIVFSP`. It sets device-side resources based on the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndexIVFSP`.<br>`int nonzeroNum`: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.<br>`int nlist`: Number of clustering centers. This corresponds to the value of the `<centroid num>` parameter in the generation of the IVFSP service operator model file.<br>`const AscendIndexIVFSP &codeBookSharedIdx`: `AscendIndexIVFSP` object that shares the codebook.<br>`faiss::ScalarQuantizer::QuantizerType qType`: Scalar quantization type. The current supported value is only `ScalarQuantizer::QuantizerType::QT_8bit`.<br>`faiss::MetricType metric`: Distance metric used by `AscendIndex` when performing feature-vector similarity retrieval. The current `faiss::MetricType metric` supports only `METRIC_L2`.<br>`AscendIndexIVFSPConfig`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | The values of `<dim>`, `<nonzero num>`, and `<centroid num>` used when training and generating the codebook must correspond to the `dims`, `nonzeroNum`, and `nlist` parameters of this function. The shared codebook configuration of `codeBookSharedIdx` must match the codebook configuration of the current index, and the device resources must also match. When `dims` ∈ {64, 128, 256}, `nlist` ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When `dims` ∈ {512, 768}, `nlist` ∈ {256, 512, 1024, 2048}. `nonzeroNum` must be a multiple of 16 and less than or equal to `min(128, dims)`. `metric` ∈ {`faiss::MetricType::METRIC_L2`}. |

<a name="table8581162710235"></a>

| API Definition | `AscendIndexIVFSP(const AscendIndexIVFSP&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor of this index as deleted. Therefore, it is a non-copyable type. |
| Input | `const AscendIndexIVFSP&`: Constant `AscendIndexIVFSP`. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table186918413239"></a>

| API Definition | `virtual ~AscendIndexIVFSP();` |
| --- | --- |
| Description | Destructor of `AscendIndexIVFSP`. It destroys the `AscendIndexIVFSP` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table241282321712"></a>

| API Definition | `AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexIVFSP`. It sets device-side resources based on the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndexIVFSP`.<br>`int nonzeroNum`: Number of nonzero dimensions after feature-vector compression and dimensionality reduction.<br>`int nlist`: Number of clustering centers. This corresponds to the value of the `<centroid num>` parameter in the generation of the IVFSP service operator model file.<br>`faiss::ScalarQuantizer::QuantizerType qType`: Scalar quantization type. The current supported value is only `ScalarQuantizer::QuantizerType::QT_8bit`.<br>`faiss::MetricType metric`: Distance metric used by `AscendIndex` when performing feature-vector similarity retrieval. The current `faiss::MetricType metric` supports only `METRIC_L2`.<br>`AscendIndexIVFSPConfig`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | When `dims` ∈ {64, 128, 256}, `nlist` ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When `dims` ∈ {512, 768}, `nlist` ∈ {256, 512, 1024, 2048}. `nonzeroNum` must be a multiple of 16 and less than or equal to `min(128, dims)`. `metric` ∈ {`faiss::MetricType::METRIC_L2`}. |

### `loadAllData API`<a id="ZH-CN_TOPIC_0000001585736172"></a>

| API Definition | `void loadAllData(const char *dataPath);` |
| --- | --- |
| Description | Load the `Index` structure from disk into the `Device`, including the compressed, reduced-dimensional feature vectors and the codebook data. |
| Input | `const char *dataPath:` Path to the data file. |
| Output | None |
| Returns | None |
| Constraints | The file corresponding to `dataPath` should be the file written by `saveAllData`, and the process user must have read permission for it. The file must not be a symbolic link.<br>This API does not support codebook sharing. If you need codebook sharing, you are advised to use the `loadAllData` overload that accepts `codeBookSharedIdx`. |

<a name="table115591219131513"></a>

| API Definition | `static std::shared_ptr<AscendIndexIVFSP> loadAllData(const AscendIndexIVFSPConfig &config, const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);` |
| --- | --- |
| Description | Restore an `AscendIndexIVFSP` object from memory. |
| Input | `const AscendIndexIVFSPConfig &config:` Device-side resource configuration. Currently, you only need to set `config.deviceList` and `config.resourceSize`. The other configuration parameters are restored from memory. `const uint8_t *data:` Memory pointer obtained by `saveAllData`. `size_t dataLen:` Actual length of the `data` pointer. `const AscendIndexIVFSP *codeBookSharedIdx:` Pointer to the `AscendIndexIVFSP` that shares the codebook. The default value is `nullptr`, which means that the codebook is not shared. |
| Output | None |
| Returns | A smart pointer to the `AscendIndexIVFSP` object restored from memory. |
| Constraints | `data` must be a non-null valid pointer. `dataLen` must be the actual length of the `data` pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The codebook configuration of the shared `codeBookSharedIdx` must match the codebook configuration of the current `Index`, and the device resources configuration must also match. |

### `operator=`<a name="ZH-CN_TOPIC_0000001635975413"></a>

| API Definition | `AscendIndexIVFSP& operator=(const AscendIndexIVFSP&) = delete;` |
| --- | --- |
| Description | Declare this `Index` assignment operator as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexIVFSP&:` A constant `AscendIndexIVFSP`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `remove_ids`<a name="ZH-CN_TOPIC_0000001635576085"></a>

| API Definition | `size_t remove_ids(const faiss::IDSelector &sel) override;` |
| --- | --- |
| Description | Implement the API for deleting the specified feature vectors from the base vector set in `AscendIndexIVFSP`. |
| Input | `const faiss::IDSelector &sel:` Feature vectors to delete. For details about the usage and definition, see the corresponding Faiss documentation. |
| Output | None |
| Returns | The number of deleted feature vectors. |
| Constraints | None |

### `reset`<a name="ZH-CN_TOPIC_0000001635815485"></a>

| API Definition | `void reset() override;` |
| --- | --- |
| Description | Clear the base vectors in this `AscendIndexIVFSP`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `saveAllData`<a name="ZH-CN_TOPIC_0000001635696053"></a>

| API Definition | `void saveAllData(const char *dataPath);` |
| --- | --- |
| Description | Write the `Index` structure from the `Device` side to disk. The data written to disk includes the compressed, reduced-dimensional feature vectors and the codebook data. |
| Input | `const char *dataPath:` Path to the output data file. |
| Output | None |
| Returns | None |
| Constraints | Ensure that the directory containing the `dataPath` file exists, and that the process user has write permission for the directory. For security hardening, the directory hierarchy must not contain symbolic links.<br>When the file corresponding to `dataPath` already exists, the file is overwritten. In this case, the process user should be the file owner. |

<a name="table11876949141314"></a>

| API Definition | `void saveAllData(uint8_t *&data, size_t &dataLen) const;` |
| --- | --- |
| Description | Store the `AscendIndexIVFSP` object in memory. |
| Input | None |
| Output | `uint8_t *&data:` Memory pointer used to store `AscendIndexIVFSP` data.<br>`size_t &dataLen:` Actual length of the `data` pointer. |
| Returns | None |
| Constraints | The input `data` must be a null pointer. After the API returns, the user must call `delete` to free the memory after using `data`. Otherwise, a memory leak occurs. |

### `search`<a name="ZH-CN_TOPIC_0000001635815489"></a>

| API Definition | `void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;` |
| --- | --- |
| Description | Implement the feature vector search API for `AscendIndexIVFSP`, and return the IDs of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n:` Number of query feature vectors.<br>`const float *x:` Feature vector data.<br>`idx_t k:` Number of most similar results to return.<br>`const SearchParameters *params:` Optional Faiss parameter. The default value is `nullptr`, and this parameter is currently unsupported. |
| Output | `float *distances:` Distance values between the query vectors and the top `k` nearest vectors. When fewer than `k` valid retrieval results are available, fill the remaining invalid distances with 65504 or `-65504`.<br>`idx_t *labels:` IDs of the top `k` nearest vectors to the query. When fewer than `k` valid retrieval results are available, fill the remaining invalid labels with `-1`. |
| Returns | None |
| Constraints | The length of the query feature vector data `x` should be `dims * n`, and the lengths of `distances` and `labels` should be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The value range of `n` here is `0 < n < 1e9`. `k` is usually not allowed to exceed 4096. |

### `search_with_filter`<a name="ZH-CN_TOPIC_0000001585736176"></a>

| API Definition | `void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const override;` |
| --- | --- |
| Description | Feature vector search API for `AscendIndexIVFSP`. It returns the IDs of the `k` most similar features based on the input feature vectors. It also provides CID-based filtering. `filters` is a `uint32_t` array with a length of `n * 6`. Every six `uint32_t` values form one filter. The first four numbers of each filter, which are 128 bits, represent the corresponding CID. The last two numbers represent the left-closed timestamp range, that is, [`x`, `y`). |
| Input | `idx_t n:` Number of query feature vectors.<br>`const float *x:` Feature vector data.<br>`idx_t k:` Number of most similar results to return.<br>`const void *filters:` Filter conditions. |
| Output | `float *distances:` Distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels:` IDs of the top `k` nearest vectors to the query. |
| Returns | None |
| Constraints | The value range of `n` is `0 < n < 1e9`. `k` is usually not allowed to exceed 4096. `x` must be a non-null pointer, and its length should be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `distances` and `labels` must be non-null pointers, and their lengths should be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `filters` must be a non-null pointer, and its length must be a `uint32_t` array of `n * 6`. Otherwise, out-of-bounds reads may occur and cause the program to crash. |

### `setNumProbes`<a name="ZH-CN_TOPIC_0000001635576089"></a>

| API Definition | `void setNumProbes(int nprobes);` |
| --- | --- |
| Description | Set the total number of candidate buckets used during search. |
| Input | `int nprobes:` `nprobe` count of `AscendIndexIVFSP`. |
| Output | None |
| Returns | None |
| Constraints | `nprobes` must be a multiple of 16 and satisfy `0 < nprobes <= nlist`. |

### `setVerbose`<a name="ZH-CN_TOPIC_0000001586055516"></a>

| API Definition | `void setVerbose(bool verbose);` |
| --- | --- |
| Description | Set whether to print the progress of adding feature vectors to the base vector set. |
| Input | `bool verbose:` Whether to print the progress of adding feature vectors to the base vector set. |
| Output | None |
| Returns | None |
| Constraints | None |

### `trainCodeBook`<a name="ZH-CN_TOPIC_0000002148530670"></a>

| API Definition | `void trainCodeBook(const AscendIndexCodeBookInitParams &codeBookInitParams) const;` |
| --- | --- |
| Description | IVFSP codebook training API. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable `OMP_NUM_THREADS=4` to speed it up. |
| Input | const AscendIndexCodeBookInitParams &codeBookInitParams: Initialization parameters required for codebook training. |
| Output | None |
| Returns | None |
| Constraints | See the `AscendIndexCodeBookInitParams` API. |

### `addCodeBook`<a name="ZH-CN_TOPIC_0000002148372594"></a>

| API Definition | `void addCodeBook(const char *codeBookPath);` |
| --- | --- |
| Description | Add a trained codebook. |
| Input | const char *codeBookPath: Codebook path. |
| Output | None |
| Returns | None |
| Constraints | The file corresponding to `codeBookPath` should be the codebook file produced by `trainCodeBook`, and the process user must have read permission for it. The file must not be a symbolic link. |

### `AscendIndexCodeBookInitParams`<a name="ZH-CN_TOPIC_0000002183731529"></a>

| API Definition | `AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum, std::string codeBookOutputDir, std::string learnDataPath, bool verbose);` |
| --- | --- |
| Description | Initialization structure for IVFSP codebook training. |
| Input | None |
| Output | None |
| Parameter Values | `int numIter:` Number of training iterations. The default value is 1.<br>`int device:` Logical device ID. The default value is 0.<br>`float ratio:` Sampling rate of the original samples used for training. The default value is `1.0`.<br>`int batchSize:` Train with batches of size `batchSize`. This value must match `<batch_size>` in the `IVFSP` training operator model file generation section. The default value is 32768.<br>`int codeNum:` Operate on at most `codeNum` samples at a time when updating the codebook. This value must be a power of two and must match `<codebook_batch_size>` in the `IVFSP` training operator model file generation section. The default value is 32768.<br>`std::string codeBookOutputDir:` Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.<br>`std::string learnDataPath:` Path to the original feature file used for training. The file supports the bin and npy formats. For bin files, the storage order is row-major and the data type is `float32`.<br>`bool verbose:` Whether to enable additional output. The default value is `true`. |
| Parameter Constraints | `numIter` ∈ (0, 20]. `ratio` ∈ (0, 1.0]. `batchSize` ∈ (0, 32768]. `codeNum` ∈ (0, 32768]. When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner. Before you run codebook training, refer to the `IVFSP` operator model file generation instructions. |

### `trainCodeBookFromMem`<a name="ZH-CN_TOPIC_0000002257319034"></a>

| API Definition | `void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &codeBookInitFromMemParams) const;` |
| --- | --- |
| Description | IVFSP codebook training API. Training data is loaded from memory. If training is slow, it may be because OpenBLAS was installed with single-threaded use restricted. You can set the environment variable `OMP_NUM_THREADS=4` to speed it up. |
| Input | const AscendIndexCodeBookInitFromMemParams &codeBookInitFromMemParams: Initialization parameters required for codebook training. |
| Output | None |
| Returns | None |
| Parameter Constraints | For details about `AscendIndexCodeBookInitFromMemParams`, see `AscendIndexCodeBookInitFromMemParams`. |

### `AscendIndexCodeBookInitFromMemParams`<a name="ZH-CN_TOPIC_0000002291969193"></a>

| API Definition | `AscendIndexCodeBookInitFromMemParams (int numIter, int device, float ratio, int batchSize, int codeNum,bool verbose,std::string codeBookOutputDir,const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd);` |
| --- | --- |
| Description | Initialization structure for IVFSP codebook training. Training data is loaded from memory. |
| Input | None |
| Output | None |
| Parameter Values | `int numIter:` Number of training iterations. The default value is 1.<br>`int device:` Logical device ID. The default value is 0.<br>`float ratio:` Sampling rate of the original samples used for training. The default value is `1.0`.<br>`int batchSize:` Train with batches of size `batchSize`. This value must match `<batch_size>` in the `IVFSP` training operator model file generation section. The value must be greater than 0, and the default value is 32768.<br>`int codeNum:` Operate on at most `codeNum` samples at a time when updating the codebook. This value must be a power of two and must match `<codebook_batch_size>` in the `IVFSP` training operator model file generation section. The value must be greater than 0, and the default value is 32768.<br>`std::string codeBookOutputDir:` Directory where the generated codebook file is written. Ensure that this directory exists and that the process user has write permission for it. For security hardening, the directory hierarchy must not contain symbolic links.<br>`bool verbose:` Whether to enable additional output. The default value is `true`.<br>`const float *memLearnData:` Pointer to in-memory data. The default value is a null pointer.<br>`size_t memLearnDataSize:` Length of the in-memory data. The default value is 0.<br>`bool isTrainAndAdd:` Whether to add the codebook directly to the `Index` after training. The default value is `false`. |
| Parameter Constraints | `numIter` ∈ (0, 20]. `ratio` ∈ (0, 1.0]. `memLearnDataSize % dim == 0`. `memLearnDataSize <= 25G`. When the codebook file already exists, it is overwritten. In this case, the process user should be the file owner. Before you run codebook training, refer to the `IVFSP` operator model file generation instructions. When `isTrainAndAdd` is `true`, the trained codebook is added directly to the `Index` and is not written to disk. When `isTrainAndAdd` is `false`, the codebook is saved to `codeBookOutputDir`, and you must call `addCodeBook` manually. `memLearnDataSize` must be the actual length of the `memLearnData` pointer. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

## `AscendIndexIVFSPConfig`<a id="ZH-CN_TOPIC_0000001635696057"></a>

`AscendIndexIVFSP` requires the corresponding `AscendIndexIVFSPConfig` to initialize the corresponding resources.

**Common Parameters<a name="section17656114673616"></a>**

| Parameter | Data Type | Parameter Description |
|--|--|--|
| handleBatch | int | Number of candidate buckets submitted for computation each time during search. The default value is 64. |
| nprobe | int | Total number of candidate buckets used during search. The default value is 64. |
| searchListSize | int | Maximum number of samples in each bucket submitted for computation each time during search. The default value is 32768. If a bucket is too large, the program automatically splits the bucket into multiple operator submissions according to `searchListSize` to compute distances. |

**API Description<a name="section74781713710"></a>**

| API Definition | `inline AscendIndexIVFSPConfig();` |
| --- | --- |
| Description | Default constructor. The default `devices` value is `{0}`, so the 0th Ascend AI Processor is used for computation. The default `resources` value is 128 MB. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table121971648373"></a>

| API Definition | `inline explicit AscendIndexIVFSPConfig(std::initializer_list<int> devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSPConfig`. It creates an `AscendIndexIVFSPConfig` and specifies the device IDs on the device side and the resource pool size. |
| Input | `std::initializer_list<int> devices:` Device-side device IDs.<br>`int64_t resources:` Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `IVF_SP_DEFAULT_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br>`uint32_t blockSize:` Preallocated memory block size, in bytes. The default value is `DEFAULT_BLOCK_SIZE` in the header file. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. Currently, only one NPU device is supported. The configured `resources` value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes. |

<a name="table56061252785"></a>

| API Definition | `inline explicit AscendIndexIVFSPConfig(std::vector<int> devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSPConfig`. It creates an `AscendIndexIVFSPConfig` and specifies the device IDs on the device side and the resource pool size. |
| Input | `std::vector<int> devices:` Device-side device IDs.<br>`int64_t resources:` Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `IVF_SP_DEFAULT_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br>`uint32_t blockSize:` Preallocated memory block size, in bytes. The default value is `DEFAULT_BLOCK_SIZE` in the header file. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. Currently, only one NPU device is supported. The configured `resources` value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes. |

## `AscendIndexIVFSQ`<a name="ZH-CN_TOPIC_0000001506334625"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456694964"></a>

The `AscendIndexIVFSQ` class uses IVF for acceleration and is a two-stage approximate retrieval algorithm.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval internally uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexIVFSQ`<a name="ZH-CN_TOPIC_0000001506414893"></a>

| API Definition | `AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQ`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexIVFScalarQuantizer *index:` CPU-side `Index`.<br>`AscendIndexIVFSQConfig config:` Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. |

<a name="table1823217151014"></a>

| API Definition | `AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQ`. It creates an `AscendIndexIVFSQ`, and the device-side resources are set according to the values configured in `config`. |
| Input | `int dims:` Dimension of the feature vectors managed by `AscendIndexIVFSQ`.<br>`int nlist:` Number of cluster centroids. This parameter corresponds to `coarse_centroid_num` in the operator generation script.<br>`faiss::ScalarQuantizer::QuantizerType qtype:` Quantizer type of `AscendIndexIVFSQ`.<br>`faiss::MetricType metric:` Distance metric used by `AscendIndex` for feature vector similarity search.<br>`bool encodeResidual:` Whether to encode residuals.<br>`AscendIndexIVFSQConfig config:` Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ {64, 128, 256, 384, 512}. `nlist` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. `qtype = ScalarQuantizer::QuantizerType::QT_8bit`, and only `ScalarQuantizer::QuantizerType::QT_8bit` is supported. `metric` ∈ {`faiss::MetricType::METRIC_L2`, `faiss::MetricType::METRIC_INNER_PRODUCT`}.<br>Note:<br>Currently, when `metric = faiss::MetricType::METRIC_INNER_PRODUCT`, `encodeResidual` only supports `false`. That is, the IVFSQ method with residual encoding is not currently supported. When `encodeResidual` is `true`, the code can run successfully, but there is an accuracy issue. |

<a name="table134501935171012"></a>

| API Definition | `AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQ`. It creates an `AscendIndexIVFSQ`, and the device-side resources are set according to the values configured in `config`. This API does not perform initialization. The subclass performs the initialization-related work. This API will be deprecated later, so do not use it. |
| Input | `int dims:` Dimension of the feature vectors managed by `AscendIndexIVFSQ`.<br>`int nlist:` Number of cluster centroids. This parameter corresponds to `coarse_centroid_num` in the operator generation script.<br>`faiss::MetricType metric:` Distance metric used by `AscendIndex` for feature vector similarity search.<br>`AscendIndexIVFSQConfig config:` Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ {64, 128, 256, 384, 512}. `nlist` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. `metric` ∈ {`faiss::MetricType::METRIC_L2`, `faiss::MetricType::METRIC_INNER_PRODUCT`}. |

<a name="table663150151113"></a>

| API Definition | `AscendIndexIVFSQ(const AscendIndexIVFSQ&) = delete;` |
| --- | --- |
| Description | Declare this `Index` copy constructor as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexIVFSQ&:` Constant `AscendIndexIVFSQ`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexIVFSQ`<a name="ZH-CN_TOPIC_0000001456534936"></a>

| API Definition | `virtual ~AscendIndexIVFSQ();` |
| --- | --- |
| Description | Destructor for `AscendIndexIVFSQ`. It destroys the `AscendIndexIVFSQ` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456375244"></a>

| API Definition | `void copyFrom(const faiss::IndexIVFScalarQuantizer *index);` |
| --- | --- |
| Description | Copy an existing `index` to Ascend based on `AscendIndexIVFSQ`, while keeping the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexIVFScalarQuantizer *index:` CPU-side `Index` resources. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer.<br>`index->d` ∈ {256}. `index->sq.d` ∈ {32, 64, 128}. The dimension of `index` must be greater than the dimension of `index->sq`, and it must be divisible by the dimension of `index->sq`. Do not call this API on an updated object. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001506334649"></a>

| API Definition | `void copyTo(faiss::IndexIVFScalarQuantizer *index) const;` |
| --- | --- |
| Description | Copy the retrieval resources of `AscendIndexIVFSQ` to the CPU side. |
| Input | `faiss::IndexIVFScalarQuantizer *index:` CPU-side `Index` resources. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The user is responsible for freeing the memory occupied by the `Index`. |

### `operator=`<a name="ZH-CN_TOPIC_0000001456854860"></a>

| API Definition | `AscendIndexIVFSQ& operator=(const AscendIndexIVFSQ&) = delete;` |
| --- | --- |
| Description | Declare this `Index` assignment operator as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexIVFSQ&:` Constant `AscendIndexIVFSQ`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `train`<a name="ZH-CN_TOPIC_0000001456854976"></a>

| API Definition | `void train(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Train `AscendIndexIVFSQ`. This class inherits the relevant APIs in `AscendIndex` and provides a concrete implementation. |
| Input | `idx_t n:` Number of feature vectors in the training set.<br>`const float *x:` Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | Training uses k-means clustering, and a small training set may affect query accuracy. The value range of `n` here is `0 < n < 1e9`. The pointer `x` must be a non-null pointer, and its length should be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

## `AscendIndexIVFSQConfig`<a id="ZH-CN_TOPIC_0000001456375204"></a>

`AscendIndexIVFSQ` requires the corresponding `AscendIndexIVFSQConfig` to initialize the corresponding resources.

**`AscendIndexIVFSQConfig`<a name="section015013311183"></a>**

| API Definition | `AscendIndexIVFSQConfig();` |
| --- | --- |
| Description | Default constructor. The default `devices` value is `{0}`, so the 0th Ascend AI Processor is used for computation. The default `resource` value is 384 MB. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table19736185071817"></a>

| API Definition | `inline AscendIndexIVFSQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQConfig`. It creates an `AscendIndexIVFSQConfig`, sets the Ascend AI Processor resources on the device side according to the values configured in `devices`, configures the resource pool size, and performs default initialization. |
| Input | `std::initializer_list<int> devices:` Device-side device IDs.<br>`int64_t resourceSize:` Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `IVFSQ_DEFAULT_TEMP_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The configured `resourceSize` value must not exceed 10 * 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes. |

<a name="table1056711401917"></a>

| API Definition | `inline AscendIndexIVFSQConfig(std::vector<int> devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQConfig`. It creates an `AscendIndexIVFSQConfig`, sets the Ascend AI Processor resources on the device side according to the values configured in `devices`, configures the resource pool size, and performs default initialization. |
| Input | `std::vector<int> devices:` Device-side device IDs.<br>`int64_t resourceSize:` Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `IVFSQ_DEFAULT_TEMP_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The configured `resourceSize` value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes. |

**`SetDefaultIVFSQConfig`<a name="section039015215286"></a>**

<a name="table1185313082915"></a>

| API Definition | `inline void SetDefaultIVFSQConfig();` |
| --- | --- |
| Description | Perform default initialization. Set the number of iterations to 16 and set a maximum of 512 points for each centroid. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendIndexIVFSQT`<a name="ZH-CN_TOPIC_0000001456375224"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506615005"></a>

The `AscendIndexIVFSQT` class contains the three-stage retrieval `IVFSQ` algorithm with dimensionality reduction. You need to pass two parameters to specify the dimensions before and after dimensionality reduction, and the original dimension must be divisible by the reduced dimension. It is suitable for scenarios with a base vector set on the order of 10 million.

You need to generate the operators required for three-stage retrieval according to the `IVFSQT` operator generation method.

This type provides fuzzy clustering. Before bucket assignment, use the `threshold` parameter to control the degree of fuzziness. Set the `threshold` value according to the base vector set capacity and the available memory size. A `threshold` that is too large can cause insufficient memory and lead to failure. For Atlas 200/300/500 inference product environments, you are advised to set it to [1.0, 1.1]. For Atlas inference series product environments, you are advised to set it to [1.0, 1.5]. For search, you are advised to use `batch size = 65536`.

The workflow is: 1. Construct the `Index` object. 2. Train the data. 3. Add the data. 4. Update the data. 5. Search the data. 6. Destroy the `Index` object. After `update`, adding data is no longer supported. If you need to search new data, destroy the original `Index` object and use the workflow again from the beginning.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval internally uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexIVFSQT`<a name="ZH-CN_TOPIC_0000001506495685"></a>

| API Definition | `AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQT`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexIVFScalarQuantizer *index:` CPU-side `Index` resources.<br>`AscendIndexIVFSQTConfig config:` Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. `index->d` ∈ {256}. `index->sq.d` ∈ {32, 64, 128}. The dimension of `index` must be greater than the dimension of `index->sq`, and it must be divisible by the dimension of `index->sq`. |

<a name="table124585216195"></a>

| API Definition | `AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT, AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFSQT`. It creates an `AscendIndexIVFSQT`, and the device-side resources are set according to the values configured in `config`. |
| Input | `int dimIn:` Dimension of the original feature vectors managed by `AscendIndexIVFSQT`.<br>`int dimOut:` Dimension of the reduced feature vectors managed by `AscendIndexIVFSQT`.<br>`int nlist:` Number of cluster centroids. This parameter corresponds to `coarse_centroid_num` in the operator generation script.<br>`faiss::ScalarQuantizer::QuantizerType qtype:` Quantizer type of `AscendIndexIVFSQT`.<br>`faiss::MetricType metric:` Distance metric used by `AscendIndex` for feature vector similarity search.<br>`AscendIndexIVFSQTConfig config:` Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dimIn` ∈ {256}. `dimOut` ∈ {32, 64, 128}. `nlist` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. `qtype = ScalarQuantizer::QuantizerType::QT_8bit`, and only the `ScalarQuantizer::QuantizerType::QT_8bit` quantizer type is supported. `metric = faiss::MetricType::METRIC_INNER_PRODUCT`, and only `faiss::MetricType::METRIC_INNER_PRODUCT` is supported. |

<a name="table68594118203"></a>

| API Definition | `AscendIndexIVFSQT(const AscendIndexIVFSQT&) = delete;` |
| --- | --- |
| Description | Declare this `Index` copy constructor as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexIVFSQT&:` `AscendIndexIVFSQT` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexIVFSQT`<a name="ZH-CN_TOPIC_0000001456854984"></a>

| API Definition | `virtual ~AscendIndexIVFSQT();` |
| --- | --- |
| Description | Destructor for `AscendIndexIVFSQT`. It destroys the `AscendIndexIVFSQT` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456695060"></a>

| API Definition | `void copyFrom(const faiss::IndexIVFScalarQuantizer *index);` |
| --- | --- |
| Description | Copy an existing `index` to Ascend based on `AscendIndexIVFSQT`, while preserving the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexIVFScalarQuantizer *index:` CPU-side `Index` resources. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer.<br>`index->d` ∈ {256}. `index->sq.d` ∈ {32, 64, 128}. The dimension of `index` must be greater than the dimension of `index->sq`, and it must be divisible by the dimension of `index->sq`. Do not call this API on an updated object. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001506495825"></a>

| API Definition | `void copyTo(faiss::IndexIVFScalarQuantizer *index);` |
| --- | --- |
| Description | Copy the retrieval resources of `AscendIndexIVFSQT` to the CPU side. |
| Input | `faiss::IndexIVFScalarQuantizer *index:` CPU-side `Index` resources. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The user frees the memory occupied by the `Index`. |

### `fineTune`<a name="ZH-CN_TOPIC_0000001456694860"></a>

| API Definition | `void fineTune(size_t n, const float *x);` |
| --- | --- |
| Description | Fine-tune and optimize the centroids to avoid uneven bucket assignment. |
| Input | `size_t n:` Number of feature vectors.<br>`const float *x:` Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `getFuzzyK`<a name="ZH-CN_TOPIC_0000001456855008"></a>

| API Definition | `int getFuzzyK() const;` |
| --- | --- |
| Description | Get the maximum value used when a vector is assigned to buckets. |
| Input | None |
| Output | None |
| Returns | `int:` Maximum value used when a vector is assigned to buckets. |
| Constraints | None |

### `getListCodesAndIds`<a name="ZH-CN_TOPIC_0000001687739112"></a>

| API Definition | `void getListCodesAndIds(int listId, std::vector<uint8_t>& codes, std::vector<ascend_idx_t>& ids) const override;` |
| --- | --- |
| Description | Return the feature vectors and corresponding IDs for a specific `nlistId` in the current `AscendIndexIVFSQT` `nlist`. |
| Input | `int listId:` Specific `nlistId` in the `nlist` of `AscendIndexIVFSQT`. |
| Output | `std::vector<uint8_t>& codes:` Feature vectors at the specific `nlistId` in the `nlist` of `AscendIndexIVFSQT`.<br>`std::vector<ascend_idx_t>& ids:` Feature vector IDs at the specific `nlistId` in the `nlist` of `AscendIndexIVFSQT`. |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `getListLength`<a name="ZH-CN_TOPIC_0000001735977797"></a>

| API Definition | `uint32_t getListLength(int listId) const override;` |
| --- | --- |
| Description | Return the length for a specific `nlistId` in the current `AscendIndexIVFSQT` `nlist`. |
| Input | `int listId:` Specific `nlistId` in the `nlist` of `AscendIndexIVFSQT`. |
| Output | None |
| Returns | Length at the specific `nlistId` in the `nlist` of `AscendIndexIVFSQT`. |
| Constraints | This API is not supported in the current version. Do not call it. |

### `getLowerBound`<a name="ZH-CN_TOPIC_0000001506614885"></a>

| API Definition | `int getLowerBound() const;` |
| --- | --- |
| Description | Return the threshold for second-level clustering. |
| Input | None |
| Output | None |
| Returns | Threshold for second-level clustering. |
| Constraints | None |

### `getMergeThres`<a name="ZH-CN_TOPIC_0000001506615073"></a>

| API Definition | `int getMergeThres() const;` |
| --- | --- |
| Description | Get the threshold for merging sub-buckets. |
| Input | None |
| Output | None |
| Returns | Threshold for merging sub-buckets. |
| Constraints | None |

### `getQMax`<a name="ZH-CN_TOPIC_0000001456535208"></a>

| API Definition | `float getQMax() const;` |
| --- | --- |
| Description | Return the maximum feature vector value. |
| Input | None |
| Output | None |
| Returns | Maximum feature vector value. |
| Constraints | None |

### `getQMin`<a name="ZH-CN_TOPIC_0000001506615029"></a>

| API Definition | `float getQMin() const;` |
| --- | --- |
| Description | Return the minimum feature vector value. |
| Input | None |
| Output | None |
| Returns | Minimum feature vector value. |
| Constraints | None |

### `getThreshold`<a name="ZH-CN_TOPIC_0000001506334633"></a>

| API Definition | `float getThreshold() const;` |
| --- | --- |
| Description | Get the threshold used to determine whether a vector is assigned to multiple buckets. |
| Input | None |
| Output | None |
| Returns | `float:` Threshold used to determine whether a vector is assigned to multiple buckets. |
| Constraints | None |

### `operator=`<a name="ZH-CN_TOPIC_0000001506615085"></a>

| API Definition | `AscendIndexIVFSQT& operator=(const AscendIndexIVFSQT&) = delete;` |
| --- | --- |
| Description | Declares the assignment operator as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexIVFSQT&`: An `AscendIndexIVFSQT` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `remove_ids`<a name="ZH-CN_TOPIC_0000001506615053"></a>

| API Definition | `size_t remove_ids(const faiss::IDSelector &sel) override;` |
| --- | --- |
| Description | Deletes base library features by ID. |
| Input | `const faiss::IDSelector &sel`: The feature vectors to delete. For details about usage and definition, see the corresponding Faiss documentation. |
| Output | None |
| Returns | The number of deleted feature vectors. |
| Constraints | This API is not supported in the current version. |

### `reset`<a name="ZH-CN_TOPIC_0000001506334789"></a>

| API Definition | `void reset() override;` |
| --- | --- |
| Description | Resets the index and clears the feature data. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | Do not continue using this object after you call this API. |

### `setAddTotal`<a name="ZH-CN_TOPIC_0000001456375316"></a>

| API Definition | `void setAddTotal(size_t addTotal);` |
| --- | --- |
| Description | Sets the total number of base library vectors to add. The default value is 100000000. You must set `PreciseMemControl` to `true` first. |
| Input | `size_t addTotal`: The total number of base library vectors to add. |
| Output | None |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `setFuzzyK`<a name="ZH-CN_TOPIC_0000001456534940"></a>

| API Definition | `void setFuzzyK(int value);` |
| --- | --- |
| Description | Sets the maximum value for each vector when it is assigned to a bucket. |
| Input | `int value`: The maximum value for each vector when it is assigned to a bucket. You are advised to keep it at the default value 3. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `value` is (0, 10]. |

### `setLowerBound`<a name="ZH-CN_TOPIC_0000001506334777"></a>

| API Definition | `void setLowerBound(int lowerBound);` |
| --- | --- |
| Description | Sets the threshold for second-level clustering. The default value is 32.<br>If the number of elements in a first-level clustering bucket is greater than `lowerBound`, second-level clustering is performed. Otherwise, the original state is retained. |
| Input | `int lowerBound`: The threshold for second-level clustering. |
| Output | None |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `setMemoryLimit`<a name="ZH-CN_TOPIC_0000001506614917"></a>

| API Definition | `void setMemoryLimit(float memoryLimit);` |
| --- | --- |
| Description | Sets the Host memory limit. The default value is 32, in `GB`. You must set `PreciseMemControl` to `true` first. |
| Input | `float memoryLimit`: The memory limit. |
| Output | None |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `setMergeThres`<a name="ZH-CN_TOPIC_0000001456694900"></a>

| API Definition | `void setMergeThres(int mergeThres);` |
| --- | --- |
| Description | Sets the threshold for merging sub-buckets. The default value is 5.<br>If the number of elements in a sub-bucket after second-level clustering is smaller than `mergeThres`, merge the elements of that sub-bucket into other sub-buckets. |
| Input | `int mergeThres`: The threshold for merging sub-buckets. |
| Output | None |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `setNumProbes`<a name="ZH-CN_TOPIC_0000001736410013"></a>

| API Definition | `void setNumProbes(int nprobes) override;` |
| --- | --- |
| Description | Sets the `nprobe` value of the current `AscendIndexIVFSQT`. |
| Input | `int nprobes`: The `nprobe` value of `AscendIndexIVFSQT`. You are advised to keep it at the default value 64. |
| Output | None |
| Returns | None |
| Constraints | `nprobes` ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. `l2Probe` ≥ `nprobes`, `l2Probe` ≤ `l3SegmentNum`, and `l2Probe` ≤ `nprobes * 64`. `l3SegmentNum` ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. For details about `l2Probe` and `l3SegmentNum`, see `setSearchParams`. `setNumProbes` is expected to be removed in September 2025. Use `setSearchParams` instead. |

### `setPreciseMemControl`<a name="ZH-CN_TOPIC_0000001506334681"></a>

| API Definition | `void setPreciseMemControl(bool preciseMemControl);` |
| --- | --- |
| Description | Specifies whether to precisely limit the memory size on the Host side. |
| Input | `bool preciseMemControl`: The default value is `false`, which disables precise memory limiting on the Host side. `true` enables it. |
| Output | None |
| Returns | None |
| Constraints | This API is not supported in the current version. Do not call it. |

### `setSearchParams`<a name="ZH-CN_TOPIC_0000002052679693"></a>

| API Definition | `void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);` |
| --- | --- |
| Description | Sets the parameters that affect retrieval accuracy and performance. |
| Input | `int nprobe`: The `nprobe` value of `AscendIndexIVFSQT`. You are advised to keep it at the default value 64.<br>`int l2Probe`: The number of sub-buckets selected during second-stage retrieval. The default value is 48.<br>`int l3SegmentNum`: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is 96. |
| Output | None |
| Returns | None |
| Constraints | `nprobe` ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. `l2Probe` ≥ `nprobe`, `l2Probe` ≤ `l3SegmentNum`, and `l2Probe` ≤ `nprobe * 64`.<br>`l3SegmentNum` ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. |

### `setSortMode`<a name="ZH-CN_TOPIC_0000002165943965"></a>

| API Definition | `void setSortMode(int mode);` |
| --- | --- |
| Description | Sets the `topk` sorting mode. Mode 0 means approximate sorting. Mode 1 means exact sorting. |
| Input | `int mode`: The `topk` sorting mode. |
| Output | None |
| Returns | None |
| Constraints | You must call this API before the `Search` API. `mode` supports only 0 or 1, and the default is 0. Mode 0: Approximate sorting truncates part of the `topk` results to improve performance. Mode 1: Exact sorting improves retrieval accuracy at the cost of some performance. |

### `setThreshold`<a name="ZH-CN_TOPIC_0000001456854808"></a>

| API Definition | `void setThreshold(float value);` |
| --- | --- |
| Description | Sets the threshold for determining whether a vector is assigned to multiple buckets. The default value is `1.0`. |
| Input | `float value`: The threshold for determining whether a vector is assigned to multiple buckets. You are advised to set it in the range [1.0, 1.5]. Because the Device side has a memory limit, once memory usage reaches the limit, the OOM mechanism is triggered and kills the process. You can check the Device-side memory limit data first (`/sys/fs/cgroup/memory/usermemory/memory.limit_in_bytes`) to estimate the size of the base library to add. If memory is tight, you are advised to keep the parameter in the range [1.0, 1.1]. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `value` is [0, `fuzzyK` - 1]. For the valid range of `fuzzyK`, see the `getFuzzyK` API. |

### `setUseCpuUpdate`<a name="ZH-CN_TOPIC_0000002167379329"></a>

| API Definition | `setUseCpuUpdate(int numThreads);` |
| --- | --- |
| Description | Specifies whether to use the CPU for update. |
| Input | `int numThreads`: The number of CPU cores used for update. The default value is the current number of CPU cores.<br>If the current CPU has more than 96 cores: if the current core count is smaller than the input `numThreads`, set `numThreads` to 96; if `96 < numThreads <=` the current core count, set `numThreads` to 96; if `numThreads <= 96`, keep the input value. If the current CPU has 96 cores or fewer: if the current core count is smaller than the input `numThreads` and `numThreads <= 96`, set `numThreads` to the current core count; if `0 < numThreads <=` the current core count, keep the input value. |
| Output | None. |
| Returns | None. |
| Constraints | The value of `numThreads` must be greater than 0. Configure it before you use `update`. |

### `train`<a name="ZH-CN_TOPIC_0000001456375352"></a>

| API Definition | `void train(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Trains `AscendIndexIVFSQT`. This class inherits the relevant APIs in `AscendIndexIVFSQ` and provides concrete implementations. |
| Input | `idx_t n`: The number of feature vectors in the training set.<br>`const float *x`: Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | Training uses k-means clustering. A training set that is too small may affect query accuracy. The valid range of `n` here is `nlist ≤ n ≤ 7,000,000`. The pointer `x` must be a non-null pointer, and its length must be `dimIn * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `update`<a name="ZH-CN_TOPIC_0000001506414869"></a>

| API Definition | `void update(bool cleanData = true);` |
| --- | --- |
| Description | This is the second stage of three-stage retrieval. After all base library data has been added and before `search` is called, this API trains sub-bucket centers and assigns vectors to buckets according to those centers. |
| Input | `cleanData`: Specifies whether to clear intermediate data. The default value is `true`. |
| Output | None |
| Returns | None |
| Constraints | You only need to call this API once in a full retrieval workflow. |

### `updateTParams`<a name="ZH-CN_TOPIC_0000001456854936"></a>

| API Definition | `void updateTParams(int l2Probe, int l3SegmentNum);` |
| --- | --- |
| Description | Passes in the parameters required for three-stage retrieval during testing. |
| Input | `int l2Probe`: The number of sub-buckets selected during second-stage retrieval. The default value is 48.<br>`int l3SegmentNum`: The number of segments processed by the L3 operator. This affects the total number of bases to search. The default value is 96. |
| Output | None |
| Returns | None |
| Constraints | `nprobe` ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}. `l2Probe` ≥ `nprobe`, `l2Probe` ≤ `l3SegmentNum`, and `l2Probe` ≤ `nprobe * 64`.<br>`l3SegmentNum` ∈ {24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020}. For details about the `nprobe` setting, see `setSearchParams`. `updateTParams` is expected to be removed in September 2026. Use `setSearchParams` instead. |

## `AscendIndexIVFSQTConfig`<a name="ZH-CN_TOPIC_0000001506495881"></a>

`AscendIndexIVFSQT` uses the corresponding `AscendIndexIVFSQTConfig` to initialize the required resources.

**AscendIndexIVFSQTConfig<a name="section6579185362314"></a>**

> [!NOTE]
> `AscendIndexIVFSQTConfig` inherits from [`AscendIndexIVFSQConfig`](#ascendindexivfsqconfig).

| API Definition | `inline AscendIndexIVFSQTConfig();` |
| --- | --- |
| Description | Default constructor. The default `devices` value is `{0}`, which uses the 0th Ascend AI Processor for computation. The default `resource` value is `384 MB`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table42413462115"></a>

| API Definition | `inline AscendIndexIVFSQTConfig(std::initializer_list<int> devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);` |
| --- | --- |
| Description | The constructor of `AscendIndexIVFSQTConfig`. It creates an `AscendIndexIVFSQTConfig` instance and, based on the values configured in `devices`, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is `IVFSQT_DEFAULT_TEMP_MEM` in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to `1024 MB` when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, non-duplicate device IDs. The configured value of `resourceSize` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). |

<a name="table0812225238"></a>

| API Definition | `inline AscendIndexIVFSQTConfig(std::vector<int> devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM);` |
| --- | --- |
| Description | The constructor of `AscendIndexIVFSQTConfig`. It creates an `AscendIndexIVFSQTConfig` instance and, based on the values configured in `devices`, sets the Ascend AI Processor resources on the Device side, configures the resource pool size, and performs the default initialization. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: The size of the pre-allocated memory pool on the Device side, in bytes. It stores intermediate results during computation and helps avoid performance fluctuations caused by dynamic memory allocation. The default parameter is `IVFSQT_DEFAULT_TEMP_MEM` in the header file. This parameter is determined by the base library size and the search batch size. You are advised to set it to `1024 MB` when the base library size is greater than or equal to 10 million and the batch size is greater than or equal to 16. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, non-duplicate device IDs. The configured value of `resourceSize` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). |

**SetDefaultIVFSQConfig<a name="section18396165022414"></a>**

<a name="table14953182017255"></a>

| API Definition | `inline void SetDefaultIVFSQConfig();` |
| --- | --- |
| Description | Performs the default initialization, sets the number of iterations to 16, and sets a maximum of 512 points for each centroid. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendIndexVStar`<a name="ZH-CN_TOPIC_0000002044351677"></a>

### Overview<a name="ZH-CN_TOPIC_0000002044510693"></a>

Ascend's self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side. It uses a self-developed matrix approximation strategy to compress feature vectors before storing them in the base library, and then uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

### `AscendIndexVStar`<a name="ZH-CN_TOPIC_0000002044513265"></a>

> [!NOTE]
>
>- When you create an `Index` instance, set `params.dim` according to the actual situation.
>- `params.subSpaceDim` and `params.nlist` should match the corresponding parameters used for codebook training.

<a name="table13851535141118"></a>

| API Definition | `explicit AscendIndexVStar(const AscendIndexVstarInitParams& params);` |
| --- | --- |
| Description | The constructor of `AscendIndexVStar`. It creates an `Index` with the corresponding dimension based on the values configured in `params`. |
| Input | `const AscendIndexVstarInitParams& params`: The constructor parameters. For details, see `AscendIndexVstarInitParams`. |
| Output | None |
| Returns | None |
| Constraints | For details, see `AscendIndexVstarInitParams`. |

<a name="table11631734281"></a>

| API Definition | `AscendIndexVStar(const std::vector<int>& deviceList, bool verbose = false);` |
| --- | --- |
| Description | The constructor of `AscendIndexVStar`. It creates an `Index` with an unknown input data dimension and unknown hyperparameters based on `deviceList`. |
| Input | `const std::vector<int>& deviceList`: Device-side device IDs.<br>`bool verbose`: Specifies whether to enable the `verbose` option. When enabled, some operations provide additional print prompts. The default value is `false`. |
| Output | None |
| Returns | None |
| Constraints | `deviceList` must contain valid device IDs. Currently, only one device is supported. After you create an `Index` instance with this constructor, you must first call `LoadIndex` to load the pre-saved `Index` instance from disk, and then you can perform other operations. |

<a name="table8937623141615"></a>

| API Definition | `AscendIndexVStar(const AscendIndexVStar&) = delete;` |
| --- | --- |
| Description | Declares this copy constructor as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexVStar&`: An `AscendIndexVStar` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `LoadIndex`<a name="ZH-CN_TOPIC_0000002008232688"></a>

<a name="table950712481817"></a>

| API Definition | `APP_ERROR LoadIndex(const std::string& indexPath, AscendIndexVStar* indexVStar = nullptr);` |
| --- | --- |
| Description | Loads an existing index from disk into the Device. |
| Input | `const std::string& indexPath`: The data file path.<br>`AscendIndexVStar* indexVStar`: Used only in the `MultiSearch` scenario so that all `Index` instances share the codebook of the first `Index` instance. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | Ensure that the directory that contains `indexPath` exists and that the user who runs the process has read permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. `indexVStar` must not be a null pointer in the `MultiSearch` scenario. It must be a null pointer in the single-`Index` scenario. If a valid `Index` pointer is used in the single-`Index` scenario, the original `Index` codebook is replaced by the codebook of the parameter `Index` instance. |

### `WriteIndex`<a name="ZH-CN_TOPIC_0000002044351681"></a>

<a name="table29774016915"></a>

| API Definition | `APP_ERROR WriteIndex(const std::string& indexPath);` |
| --- | --- |
| Description | Writes the index to disk. |
| Input | `const std::string& indexPath`: The file path where the data is saved. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | Ensure that the directory that contains `indexPath` exists and that the user who runs the process has write permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. If the file already exists, it is overwritten. In this case, the user who runs the process must be the owner of the file. |

### `AddCodeBooksByIndex`<a name="ZH-CN_TOPIC_0000002044510697"></a>

<a name="table81089131197"></a>

| API Definition | `APP_ERROR AddCodeBooksByIndex(AscendIndexVStar& indexVStar);` |
| --- | --- |
| Description | In a multi-`Index` retrieval scenario, this API loads the codebook of the input `Index` instance into the current `Index`. |
| Input | `AscendIndexVStar& indexVStar`: An `Index` instance with the codebook already populated. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | This API is used only in the `MultiSearch` scenario. |

### `AddCodeBooksByPath`<a name="ZH-CN_TOPIC_0000002008390980"></a>

<a name="table1523424814919"></a>

| API Definition | `APP_ERROR AddCodeBooksByPath(const std::string& codeBooksPath);` |
| --- | --- |
| Description | Loads a codebook into the current `Index` from the codebook path. |
| Input | `const std::string& codeBooksPath`: The codebook data file path. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | Ensure that the directory that contains `codeBooksPath` exists and that the user who runs the process has read permission on the directory. For security hardening, the directory hierarchy cannot contain symbolic links. |

### `Add`<a name="ZH-CN_TOPIC_0000002008232692"></a>

<a name="table18288921121213"></a>

| API Definition | `APP_ERROR Add(const std::vector<float>& baseData);` |
| --- | --- |
| Description | Builds the `AscendIndexVStar` base library and adds new feature vectors to the base library. |
| Input | `const std::vector<float>& baseData`: The feature vectors to add to the base library. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | The length of `baseData` must be `n * dim`, where `n` is the number of vectors to add to the base library and `dim` is the vector dimension. `n` must be in the range [10000, 1e8].<br>This API does not set IDs. The default ID range of the base library is [`ntotal`, `ntotal` + `n`), where `ntotal` is the number of vectors already in the `Index`, and `n` is the number of vectors to add to the base library. |

> [!NOTE]
>
>- The `Add` API cannot be used together with the `AddWithIds` API.
>- After you use the `Add` API, the labels in the `Search` results may be duplicated. If your business logic requires labels, you are advised to use the [AddWithIds API](#addwithids).

### `AddWithIds`<a name="ZH-CN_TOPIC_0000002044351685"></a>

<a name="table32483414124"></a>

| API Definition | `APP_ERROR AddWithIds(const std::vector<float>& baseData, const std::vector<int64_t>& ids);` |
| --- | --- |
| Description | Builds the `AscendIndexVStar` base library and adds new feature vectors to the base library. This API allows the user to specify the IDs of the base library vectors to add. |
| Input | `const std::vector<float>& baseData`: The feature vectors to add to the base library.<br>`const std::vector<int64_t>& ids`: The array of IDs to map to the base library vectors to add. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | The length of `baseData` must be `n * dim`, where `n` is the number of vectors to add to the base library and `dim` is the vector dimension. The length of `ids` must be `n`. Based on your own business scenario, ensure that `ids` are valid. If duplicate IDs exist in the base library, the `label` in the retrieval results cannot correspond to a specific base library vector. `n` must be in the range [10000, 1e8]. |

### `DeleteByIds`<a name="ZH-CN_TOPIC_0000002044510701"></a>

<a name="table1284884631210"></a>

| API Definition | `APP_ERROR DeleteByIds(const std::vector<int64_t>& ids);` |
| --- | --- |
| Description | Deletes the vector data in the base library that corresponds to the IDs in the parameter array. |
| Input | `const std::vector<int64_t>& ids`: The array of vector IDs to delete from the base library. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | The IDs in `ids` must be IDs used by the base library addition API. |

### `DeleteById`<a name="ZH-CN_TOPIC_0000002008390984"></a>

<a name="table9845165841212"></a>

| API Definition | `APP_ERROR DeleteById(int64_t id);` |
| --- | --- |
| Description | Deletes the vector data in the base library that corresponds to the parameter ID. |
| Input | `int64_t id`: The ID of the base library vector to delete. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | The ID must be an ID used by the base library addition API. |

### `DeleteByRange`<a name="ZH-CN_TOPIC_0000002008232696"></a>

<a name="table103969158136"></a>

| API Definition | `APP_ERROR DeleteByRange(int64_t startId, int64_t endId);` |
| --- | --- |
| Description | Deletes the vector data in the base library that corresponds to the ID range in the parameters. |
| Input | `int64_t startId`: The starting ID of the base library vectors to delete.<br>`int64_t endId`: The ending ID of the base library vectors to delete. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | The IDs to delete must be IDs used by the base library addition API, and the ID must be in the range [`startId`, `endId`]. |

### `Search`<a name="ZH-CN_TOPIC_0000002044351689"></a>

<a name="table197566920146"></a>

| API Definition | `APP_ERROR Search(const AscendIndexSearchParams& params) const;` |
| --- | --- |
| Description | Performs feature vector retrieval and returns the IDs of the most similar `topK` features based on the input feature vectors. |
| Input | `const AscendIndexSearchParams& params`: The retrieval parameters. For details, see `AscendIndexSearchParams`.<br>`size_t n`: The number of query feature vectors.<br>`std::vector<float>& queryData`: Feature vector data.<br>`int topK`: The number of most similar results to return. |
| Output | `std::vector<float>& dists`: The distance values between the query vectors and the closest `topK` vectors.<br>`std::vector<int64_t>& labels`: The IDs of the closest `topK` vectors. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | `n` ∈ (0, 10000]. Ensure that `n * dim * sizeof(float)` is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. `queryData`: The length must be greater than or equal to `n * dim`. `topK` ∈ (0, 4096]. `dists` and `labels`: The length must be greater than or equal to `n * topK`. |

### `SearchWithMask`<a name="ZH-CN_TOPIC_0000002044510705"></a>

<a name="table777072291418"></a>

| API Definition | `APP_ERROR SearchWithMask(const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask) const;` |
| --- | --- |
| Description | Performs feature vector retrieval and returns the IDs of the most similar `topK` features based on the input feature vectors. `mask` is a 0 and 1 bit string. Each bit indicates whether the corresponding feature in the base library participates in distance calculation. 0 means it does not participate, and 1 means it does. |
| Input | `const AscendIndexSearchParams& params`: The retrieval parameters. For details, see `AscendIndexSearchParams`.<br>`size_t n`: The number of query feature vectors.<br>`std::vector<float>& queryData`: Feature vector data.<br>`int topK`: The number of most similar results to return.<br>`const std::vector<uint8_t>& mask`: The feature base library mask. |
| Output | `std::vector<float>& dists`: The distance values between the query vectors and the closest `topK` vectors.<br>`std::vector<int64_t>& labels`: The IDs of the closest `topK` vectors. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | `n` ∈ (0, 10000]. Ensure that `n * dim * sizeof(float)` is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. `queryData`: The length must be greater than or equal to `n * dim`. `topK` ∈ (0, 4096]. `dists` and `labels`: The length must be greater than or equal to `n * topK`. `mask`: The length must be greater than or equal to `n * ceil(ntotal/8)`, where `ntotal` is the number of base library features. |

### `MultiSearch`<a name="ZH-CN_TOPIC_0000002008390988"></a>

<a name="table158666394146"></a>

| API Definition | `APP_ERROR MultiSearch(std::vector<AscendIndexVStar*>& indexes, const AscendIndexSearchParams& params, bool merge) const;` |
| --- | --- |
| Description | Performs feature vector retrieval across multiple `AscendIndexVStar` libraries and returns the IDs and distances of the most similar `topK` features based on the input feature vectors. |
| Input | `std::vector<AscendIndexVStar*>& indexes`: Multiple `Index` instances to search.<br>`const AscendIndexSearchParams& params`: The retrieval parameters. For details, see `AscendIndexSearchParams`.<br>`size_t n`: The number of query feature vectors.<br>`std::vector<float>& queryData`: Feature vector data.<br>`int topK`: The number of most similar results to return.<br>`bool merge`: Specifies whether to merge the retrieval results across multiple `Index` instances. |
| Output | `std::vector<float>& dists`: The distance values between the query vectors and the closest `topK` vectors.<br>`std::vector<int64_t>& labels`: The IDs of the closest `topK` vectors. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | `n` ∈ (0, 10000]. Ensure that `n * dim * sizeof(float)` is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. `queryData`: The length must be greater than or equal to `n * dim`. `topK` ∈ (0, 4096]. `dists` and `labels` must meet the following requirements. When `merge = true`, the length must be greater than or equal to `n * topK`. When `merge = false`, the length must be greater than or equal to `indexes.size() * n * topK`. `indexes` must meet the following requirement: `0 < indexes.size() ≤ 150`. |

### `MultiSearchWithMask`<a name="ZH-CN_TOPIC_0000002008232700"></a>

<a name="table141672058131413"></a>

| API Definition | `APP_ERROR MultiSearchWithMask(std::vector<AscendIndexVStar*>& indexes, const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask, bool merge);` |
| --- | --- |
| Description | Performs feature vector retrieval across multiple `AscendIndexVStar` libraries and returns the IDs and distances of the most similar `topK` features based on the input feature vectors. It also supports deciding whether the base library participates in distance calculation based on a `mask`. `mask` is a 0 and 1 bit string. Each bit indicates whether the corresponding feature in the base library participates in distance calculation. 0 means it does not participate, and 1 means it does. |
| Input | `std::vector<AscendIndexVStar*>& indexes`: Multiple `Index` instances to search.<br>`const AscendIndexSearchParams& params`: The retrieval parameters. For details, see `AscendIndexSearchParams`.<br>`size_t n`: The number of query feature vectors.<br>`std::vector<float>& queryData`: Feature vector data.<br>`int topK`: The number of most similar results to return.<br>`const std::vector<uint8_t>& mask`: The feature base library mask.<br>`bool merge`: Specifies whether to merge the retrieval results across multiple `Index` instances. |
| Output | `std::vector<float>& dists`: The distance values between the query vectors and the closest `topK` vectors.<br>`std::vector<int64_t>& labels`: The IDs of the closest `topK` vectors. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | `n` ∈ (0, 10000]. Ensure that `n * dim * sizeof(float)` is smaller than the remaining memory on the card. Otherwise, insufficient memory may cause retrieval to fail. `queryData`: The length must be greater than or equal to `n * dim`. `topK` ∈ (0, 4096]. `dists` and `labels` must meet the following requirements. When `merge = true`, the length must be greater than or equal to `n * topK`. When `merge = false`, the length must be greater than or equal to `indexes.size() * n * topK`. `mask`: The length must be greater than or equal to `n * ceil(ntotal_max/8)`, where `ntotal_max` is the number of base library features and is the maximum number of base library vectors among all `Index` instances. `indexes` must meet the following requirement: `0 < indexes.size() ≤ 150`. |

### `SetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002044351693"></a>

<a name="table4215111781514"></a>

| API Definition | `APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams& params);` |
| --- | --- |
| Description | Sets the hyperparameters used when an `AscendIndexVstar` instance performs retrieval. |
| Input | `const AscendIndexVstarHyperParams& params`: The retrieval hyperparameters. For details, see `AscendIndexVstarHyperParams`. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | `nProbeL1` ∈ (16, `nListL1`], `nProbeL1 % 8 == 0`. `nProbeL2` ∈ (16, `nProbeL1` * `nList2`], `nProbeL2 % 8 == 0`. `l3SegmentNum` ∈ (100, 5000], `l3SegmentNum % 8 == 0`. |

### `GetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002044510709"></a>

<a name="table5860202961515"></a>

| API Definition | `APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams& params) const;` |
| --- | --- |
| Description | Gets the hyperparameters used during vector retrieval. |
| Input | None |
| Output | `AscendIndexVstarHyperParams& params`: The retrieval hyperparameters. For details, see `AscendIndexVstarHyperParams`. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | None |

### `GetDim`<a name="ZH-CN_TOPIC_0000002008390992"></a>

<a name="table6661184351519"></a>

| API Definition | `APP_ERROR GetDim(int& dim) const;` |
| --- | --- |
| Description | Gets the dimension used when the index is initialized. |
| Input | None |
| Output | `int& dim`: The dimension of the `Index`. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | None |

### `GetNTotal`<a name="ZH-CN_TOPIC_0000002008232704"></a>

<a name="table1919613597154"></a>

| API Definition | `APP_ERROR GetNTotal(uint64_t& ntotal) const;` |
| --- | --- |
| Description | Gets the number of base library vectors in the current index. |
| Input | None |
| Output | `uint64_t& ntotal`: The total number of base library vectors in the current `Index`. |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | None |

### `Reset`<a name="ZH-CN_TOPIC_0000002044351697"></a>

<a name="table19794117167"></a>

| API Definition | `APP_ERROR Reset();` |
| --- | --- |
| Description | Resets the index and clears the saved index data. |
| Input | None |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | After you reset the index, the parameters that the user provided when initializing the index are retained. |

### `operator=`<a name="ZH-CN_TOPIC_0000002008390996"></a>

<a name="table3792193711620"></a>

| API Definition | `AscendIndexVStar& operator=(const AscendIndexVStar&) = delete;` |
| --- | --- |
| Description | Declares the assignment operator as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexVStar&`: An `AscendIndexVStar` object. |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendIndexGreat`<a name="ZH-CN_TOPIC_0000002044829945"></a>

### Overview<a name="ZH-CN_TOPIC_0000002008751966"></a>

This self-developed vector retrieval algorithm provides approximate retrieval for high-dimensional large base libraries on the Ascend side and the Kunpeng side. It uses a self-developed retrieval strategy to retrieve the top `K` most similar vectors from the base library.

The vectors stored in the base library and the query vectors passed to each API must be normalized `float` values.

This algorithm does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you must acquire a lock before use. Otherwise, the retrieval API may fail. It also does not support sharing one Device across threads.

This algorithm is mainly designed for approximate fuzzy search in large base library scenarios, and its accuracy is lower than brute-force retrieval. In small base library scenarios, you are advised to increase the hyperparameter values appropriately to reduce the loss in accuracy.

> [!NOTE]
>
>- When you create an `Index` instance, set `params.dim` according to the actual situation.
>- The `Index` has two algorithm modes: `KMode`, which uses only the Kunpeng-side algorithm, and `AKMode`, which uses the Ascend plus Kunpeng algorithm. In `AKMode`, you must generate the corresponding operators in advance.
>- Ensure that `subSpaceDim` and `nlist` match the corresponding parameters used for codebook training.

### `AscendIndexGreat`<a name="ZH-CN_TOPIC_0000002044829953"></a>

<a name="table5404639201712"></a>

| API Definition | `AscendIndexGreat(const std::string& mode, const std::vector<int>& deviceList, bool verbose = false);` |
| --- | --- |
| Description | The constructor of `AscendIndexGreat`. It creates a retrieval `Index` on Ascend. |
| Input | `const std::string& mode`: Specifies the algorithm mode.<br>`const std::vector<int>& deviceList`: The specified NPU-side device IDs.<br>`bool verbose`: Specifies whether to enable the `verbose` option. When enabled, some operations provide additional print prompts. The default value is `false`. |
| Output | None |
| Returns | None |
| Constraints | `mode` supports only `KMode` and `AKMode`. For `deviceList`, use the `npu-smi` command to query the corresponding NPU IDs. Only one device ID is supported. After you create an `Index` instance with this constructor, you must first call `LoadIndex` to load the pre-saved `Index` instance from disk, and then you can perform other operations. |

<a name="table72261454131719"></a>

| API Definition | `explicit AscendIndexGreat(const AscendIndexGreatInitParams& kModeInitParams);` |
| --- | --- |
| Description | The constructor of `AscendIndexGreat`. It creates a retrieval `Index` on Ascend. |
| Input | Initialization parameters required by the `Index`, specifically `kModeInitParams`. For details, see `AscendIndexGreatInitParams`. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | See the parameter descriptions and constraints in `AscendIndexGreatInitParams`. |

<a name="table198261931819"></a>

| API Definition | `AscendIndexGreat(const AscendIndexVstarInitParams& aModeInitParams, const AscendIndexGreatInitParams& kModeInitParams);` |
| --- | --- |
| Description | The constructor of `AscendIndexGreat`. It creates a retrieval `Index` on Ascend. |
| Input | Initialization parameters required by the `Index`, specifically `aModeInitParams` and `kModeInitParams`. For details, see `AscendIndexVstarInitParams` and `AscendIndexGreatInitParams`. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | Refer to the parameter descriptions and constraints in `AscendIndexVstarInitParams` and `AscendIndexGreatInitParams`.<br>The `dim` values of `aModeInitParams` and `kModeInitParams` must be the same. |

<a name="table32891532172215"></a>

| API Definition | `AscendIndexGreat(const AscendIndexGreat&) = delete;` |
| --- | --- |
| Description | Declares this copy constructor as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexGreat&`: A constant `AscendIndexGreat` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexGreat`<a name="ZH-CN_TOPIC_0000002013257524"></a>

| API Definition | `virtual ~AscendIndexGreat() = default;` |
| --- | --- |
| Description | The destructor of `AscendIndexGreat`. It destroys the `AscendIndexGreat` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `operator=`<a name="ZH-CN_TOPIC_0000002008751990"></a>

<a name="table39961720122213"></a>

| API Definition | `AscendIndexGreat &operator=(const AscendIndexGreat&) = delete;` |
| --- | --- |
| Description | Declares the assignment operator as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexGreat&`: A constant `AscendIndexGreat` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `Add`<a name="ZH-CN_TOPIC_0000002044950953"></a>

<a name="table11133547191811"></a>

| API Definition | `APP_ERROR Add(const std::vector<float>& baseRawData);` |
| --- | --- |
| Description | Adds new feature vectors to the `AscendIndexGreat` base library. |
| Input | `const std::vector<float>& baseRawData`: The feature vectors to add to the base library. |
| Output | None |
| Returns | `APP_ERROR`: The operation status. For details, see the API return value reference. |
| Constraints | The length of `baseRawData` must be `dim * nTotal`. `nTotal` is the number of vectors to add to the base library, and `dim` is the dimension of each vector. The valid range of the total number of base library vectors is `10000 ≤ nTotal ≤ 1e8`. This algorithm does not support adding data again after the base library has been added. The `Add` API cannot be used together with the `AddWithIds` API. |

### `AddWithIds`<a name="ZH-CN_TOPIC_0000002044829957"></a>

<a name="table2436200181918"></a>

| API Definition | `APP_ERROR AddWithIds (const std::vector<float>& baseRawData, const std::vector<int64_t>& ids);` |
| --- | --- |
| Description | Adds new feature vectors to the AscendIndexGreat base index. When features are added through `AddWithIds`, the default IDs for the corresponding features are [0, `ntotal`). |
| `Input` | `const std::vector<float>& baseRawData`: Feature vectors to add to the base index.<br>`const std::vector<int64_t>& ids`: IDs of the feature vectors to add to the base index. IDs must be unique within the `Index` instance. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | The length of the `baseRawData` array must be `dim * nTotal`. `nTotal` is the number of vectors to be added to the base index, and `dim` is the dimensionality of each vector. The total number of base vectors must satisfy `10000 ≤ nTotal ≤ 1e8`. The length of `ids` must be `nTotal`. Users must ensure the validity of `ids` according to their own business scenario. If duplicate IDs exist in the base index, the `label` in the search results cannot be mapped to a specific base vector. This algorithm does not support adding vectors after the base index has been built. The `AddWithIds` API cannot be used together with the `Add` API. |

### `LoadIndex`<a name="ZH-CN_TOPIC_0000002008751978"></a>

<a name="table17789162191912"></a>

| API Definition | `APP_ERROR LoadIndex(const std::string& indexPath);` |
| --- | --- |
| Description | Loads the `Index` structure from disk, including compressed, dimension-reduced feature vectors and codebook data. |
| `Input` | `const std::string& indexPath`: Path to load the KMode index. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | The file corresponding to `indexPath` must be a persisted file generated by calling `WriteIndex`, and the running user must have read permission for it. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy. |

<a name="table98570373191"></a>

| API Definition | `APP_ERROR LoadIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath);` |
| --- | --- |
| Description | Writes the `Index` structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and the original data. |
| `Input` | `const std::string& aModeIndexPath`: Path to load the AMode index.<br>`const std::string& kModeIndexPath`: Path to load the KMode index. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | The files corresponding to `aModeIndexPath` and `kModeIndexPath` must be the persisted files generated by calling `WriteIndex`, and the running user must have read permission for them. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy. |

### `WriteIndex`<a name="ZH-CN_TOPIC_0000002044950957"></a>

<a name="table84194504191"></a>

| API Definition | `APP_ERROR WriteIndex(const std::string& indexPath);` |
| --- | --- |
| Description | Writes the `Index` structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data. |
| `Input` | None |
| Output | `const std::string& indexPath`: Path to write the KMode index. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | The user must ensure that the directory containing the `indexPath` file exists and that the running user has write permission for that directory. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy. |

<a name="table14392122132014"></a>

| API Definition | `APP_ERROR WriteIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath);` |
| --- | --- |
| Description | Writes the `Index` structure to disk. The data written to disk includes compressed, dimension-reduced feature vectors and codebook data. |
| `Input` | None |
| Output | `const std::string& aModeIndexPath`: Path to write the AMode index.<br>`const std::string& kModeIndexPath`: Path to write the KMode index. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | The user must ensure that the directories containing the `aModeIndexPath` and `kModeIndexPath` file paths exist and that the running user has write permission for those directories. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy. |

### `AddCodeBooks`<a name="ZH-CN_TOPIC_0000002008751982"></a>

<a name="table339181620207"></a>

| API Definition | `APP_ERROR AddCodeBooks(const std::string& codeBooksPath);` |
| --- | --- |
| Description | Loads an already generated codebook into the `Index`. |
| `Input` | `const std::string& codeBooksPath`: Path to the generated codebook. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | This API can only be used when initializing an index in `AKMode`.<br>The user must ensure that the directory containing the `codeBooksPath` file exists, and the file content must be a valid codebook. For security hardening, no symbolic links are allowed anywhere in the directory hierarchy. |

### `Search`<a name="ZH-CN_TOPIC_0000002008910274"></a>

<a name="table537563852013"></a>

| API Definition | `APP_ERROR Search(const AscendIndexSearchParams& searchParams);` |
| --- | --- |
| Description | Implements the AscendIndexGreat feature-vector search API. Based on the input feature vectors, it returns the distances and IDs of the most similar `topK` features. |
| `Input` | For the `searchParams` structure, see the `AscendIndexSearchParams` API.<br>`size_t n`: Number of query feature vectors.<br>`std::vector<float>& queryData`: Feature-vector data.<br>`int topK`: Number of most similar results to return. |
| Output | `std::vector<float>& dists`: Distance values between the query vectors and the top `topK` nearest vectors.<br>`std::vector<int64_t>& labels`: IDs of the top `topK` nearest vectors to the query. When the number of valid search results is less than `topK`, the remaining invalid labels are filled with `-1`. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `topK` ∈ (0, 4096]. `n` ∈ (0, 10000]. `queryData` cannot be empty, and its data length must be greater than or equal to `n * dim`. `dists` cannot be empty, and the length of the data it points to must be greater than or equal to `n * topK`. `labels` cannot be empty, and the length of the data it points to must be greater than or equal to `n * topK`. |

### `SearchWithMask`<a name="ZH-CN_TOPIC_0000002044950961"></a>

<a name="table186956182018"></a>

| API Definition | `APP_ERROR SearchWithMask(const AscendIndexSearchParams& searchParams, const std::vector<uint8_t>& mask);` |
| --- | --- |
| Description | Implements the AscendIndexGreat feature-vector search API. Based on the input feature vectors, it returns the distances and IDs of the most similar `topK` features. In addition, the user can input a `uint8` array to mask specific base-index IDs so that the feature vectors corresponding to those IDs are excluded from retrieval. |
| `Input` | For the `searchParams` structure, see the `AscendIndexSearchParams` API.<br>`size_t n`: Number of query feature vectors.<br>`std::vector<float>& queryData`: Feature-vector data.<br>`int topK`: Number of most similar results to return.<br>`const std::vector<uint8_t>& mask`: External filtering mask, in bits. 0 means the feature is filtered out; 1 means the feature is selected. |
| Output | `std::vector<float>& dists`: Distance values between the query vectors and the top `topK` nearest vectors.<br>`std::vector<int64_t>& labels`: IDs of the top `topK` nearest vectors to the query. When the number of valid search results is less than `topK`, the remaining invalid labels are filled with `-1`. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `topK` ∈ (0, 4096]. `n` ∈ (0, 10000]. `queryData` cannot be empty, and its data length must be greater than or equal to `n * dim`. `dists` cannot be empty, and the length of the data it points to must be greater than or equal to `n * topK`. `labels` cannot be empty, and the length of the data it points to must be greater than or equal to `n * topK`. The total amount of data pointed to by `mask` must be greater than or equal to `n * ceil(nTotal / 8)`. |

### `GetNTotal`<a name="ZH-CN_TOPIC_0000002044829965"></a>

<a name="table971712872115"></a>

| API Definition | `APP_ERROR GetNTotal (uint64_t& nTotal) const;` |
| --- | --- |
| Description | Gets the number of feature vectors that have been added to the AscendIndexGreat base index. |
| `Input` | None |
| Output | `uint64_t& nTotal`: Number of feature vectors added to the base index. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | None |

### `GetDim`<a name="ZH-CN_TOPIC_0000002008751986"></a>

<a name="table113422226216"></a>

| API Definition | `APP_ERROR GetDim(int& dim) const;` |
| --- | --- |
| Description | Gets the dimensionality of the feature vectors added to the AscendIndexGreat base index. |
| `Input` | None |
| Output | `int& dim`: Dimensionality of the feature vectors added to the base index. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | None |

### `Reset`<a name="ZH-CN_TOPIC_0000002008910278"></a>

<a name="table1974793512118"></a>

| API Definition | `APP_ERROR Reset();` |
| --- | --- |
| Description | Clears the data stored in this `Index`, including compressed, dimension-reduced feature vectors and codebook data, while retaining the parameters entered when the user initialized the index. |
| `Input` | None |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | None |

### `SetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002044950965"></a>

<a name="table1011347192118"></a>

| API Definition | `APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams& params);` |
| --- | --- |
| Description | Sets the hyperparameters used when searching this `Index`. |
| `Input` | `const AscendIndexHyperParams& params`: Search hyperparameters. For details, see `AscendIndexHyperParams`. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | None |

### `GetHyperSearchParams`<a name="ZH-CN_TOPIC_0000002400547905"></a>

<a name="table749915518225"></a>

| API Definition | `APP_ERROR GetHyperSearchParams(AscendIndexHyperParams& params) const;` |
| --- | --- |
| Description | Gets the search hyperparameters used when searching this `Index`. |
| `Input` | None |
| Output | `AscendIndexHyperParams& params`: Search hyperparameters. For details, see `AscendIndexHyperParams`. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | None |

## `AscendIndexMixSearchParams`<a name="ZH-CN_TOPIC_0000002008910258"></a>

### Overview<a name="ZH-CN_TOPIC_0000002045034929"></a>

The `AscendIndexMixSearchParams.h` file provides the structures required by `AscendIndexGreat` and `AscendIndexVStar`.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must acquire a lock before use; otherwise, the search API may cause exceptions. Sharing a single device across different threads is also not supported.

### `AscendIndexGreatInitParams`<a name="ZH-CN_TOPIC_0000002049404289"></a>

<a name="table17465519101616"></a>

| API Definition | `AscendIndexGreatInitParams();` |
| --- | --- |
| Description | Initialization parameter structure for KMode mode. |
| `Input` | None |
| Output | None |
| Constraints | See `AscendIndexGreatInitParams` for default parameter values. |

<a id="table10419189143817"></a>

| API Definition | `AscendIndexGreatInitParams(int dim, int degree, int convPQM, int evaluationType, int expandingFactor);` |
| --- | --- |
| Description | Initialization parameter structure for KMode mode. |
| `Input` | `int dim`: Dimensionality of the feature vectors.<br>`int degree`: Controls the fineness of the graph index during index construction. A larger value makes the graph index more fine-grained, requires more space, and yields higher retrieval accuracy.<br>`int convPQM`: Number of PQ quantization vector segments.<br>`int evaluationType`: Distance evaluation algorithm type; 0 represents IP and 1 represents L2.<br>`int expandingFactor`: Number of neighbors connected when searching each layer during the initial graph-construction phase. Note that this is different from the retrieval-stage `expandingFactor`. |
| Output | None |
| Constraints | `dim` ∈ {128, 256, 512, 1024}, default value: 256. `degree` ∈ [50, 100], default value: 50. `convPQM` must be at least 16, must be a multiple of 8, and must be divisible by `dim`; default value: 128. `evaluationType` ∈ {0, 1}, default value: 0. `expandingFactor` ∈ [200, 400], must be a multiple of 10; default value: 300. |

### `AscendIndexVstarInitParams`<a name="ZH-CN_TOPIC_0000002013246410"></a>

<a name="table20955195613391"></a>

| API Definition | `AscendIndexVstarInitParams();` |
| --- | --- |
| Description | Initialization parameter structure for Vstar mode. |
| `Input` | None |
| Output | None |
| Constraints | See `AscendIndexVstarHyperParams` for default parameter values. |

<a id="table899624214019"></a>

| API Definition | `AscendIndexVstarInitParams(int dim, int subSpaceDim, int nlist, const std::vector<int>& deviceList, bool verbose = false, int64_t resourceSize = VSTAR_DEFAULT_MEM);` |
| --- | --- |
| Description | Initialization parameter structure for Vstar mode. |
| `Input` | `int dim`: Dimensionality of the feature vectors.<br>`int subSpaceDim`: Dimensionality after the first dimensionality reduction.<br>`int nlist`: Number of first-level clusters.<br>`const std::vector<int>& deviceList`: Specified NPU physical IDs.<br>`bool verbose`: Whether to enable the `verbose` option. When enabled, some operations provide additional printed messages. Default value: `false`.<br>`int64_t resourceSize`: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is `VSTAR_DEFAULT_MEM` defined in the header file, with a size of 128 MB. This parameter is determined jointly by the base index size and the `search` batch size. |
| Output | None |
| Constraints | `dim` ∈ {128, 256, 512, 1024}, default value: 1024.<br>`subSpaceDim` ∈ {32, 64, 128}. `subSpaceDim` must be less than `dim`. Default value: 128.<br>`nlist` ∈ {256, 512, 1024}. Default value: 1024.<br>For `deviceList`, use the `npu-smi` command to query the physical ID of the corresponding NPU card. Only one device ID is supported.<br>`resourceSize` ∈ [128M, 2048M]. |

### `AscendIndexVstarHyperParams`<a name="ZH-CN_TOPIC_0000002013404694"></a>

<a name="table201855541164"></a>

| API Definition | `AscendIndexVstarHyperParams();` |
| --- | --- |
| Description | Hyperparameter structure for VSTAR mode. |
| `Input` | None |
| Output | None |
| Constraints | See `AscendIndexVstarHyperParams` for default parameter values. |

<a id="table42921559204019"></a>

| API Definition | `AscendIndexVstarHyperParams(int nProbeL1, int nProbeL2, int l3SegmentNum);` |
| --- | --- |
| Description | Hyperparameter structure for VSTAR mode. |
| `Input` | `int nProbeL1`: Number of clusters searched in the first-stage retrieval.<br>`int nProbeL2`: Number of clusters searched in the second-stage retrieval.<br>`int l3SegmentNum`: Number of segments in the third-stage retrieval, that is, the number of data segments searched from `nProbeL2`. |
| Output | None |
| Constraints | `nProbeL1` ∈ [32, `nListL1`], and `nProbeL1` must be an integer multiple of 8. Default value: 72. `nProbeL2` ∈ (16, `nProbeL1` * `n`]; when `dim` is 1024, `n` is 16, and for other dimensions `n` is 32. `nProbeL2` must be an integer multiple of 8. Default value: 64. `l3SegmentNum` ∈ (100, 5000], and `l3SegmentNum` must be an integer multiple of 8. Default value: 512. |

### `AscendIndexHyperParams`<a name="ZH-CN_TOPIC_0000002049325253"></a>

<a name="table93967711712"></a>

| API Definition | `AscendIndexHyperParams();` |
| --- | --- |
| Description | Hyperparameter structure for GREAT retrieval. |
| `Input` | None |
| Output | None |
| Constraints | See `AscendIndexHyperParams` for default parameter values. |

<a id="table1334182412417"></a>

| API Definition | `AscendIndexHyperParams(const std::string& mode, const AscendIndexVstarHyperParams& vstarHyperParam, int expandingFactor);` |
| --- | --- |
| Description | Hyperparameter structure for GREAT retrieval. |
| `Input` | `const std::string& mode`: Specifies the algorithm mode.<br>`const AscendIndexVstarHyperParams& vstarHyperParam`: For details, see `AscendIndexVstarHyperParams`.<br>`int expandingFactor`: Number of neighbors searched at each layer during retrieval. Note that this differs from the `expandingFactor` used during graph construction. |
| Output | None |
| Constraints | `mode` ∈ {"KMode", "AKMode"}. Default value: `AKMode`. `expandingFactor` ∈ [10, 200]. Default value: 150. |

<a name="table88027219236"></a>

| API Definition | `AscendIndexHyperParams(const std::string& mode, int expandingFactor);` |
| --- | --- |
| Description | Hyperparameter structure for GREAT retrieval. |
| `Input` | `const std::string& mode`: Specifies the algorithm mode.<br>`int expandingFactor`: Number of neighbors searched at each layer during retrieval. Note that this differs from the `expandingFactor` used during graph construction. |
| Output | None |
| Constraints | `mode` ∈ {"KMode", "AKMode"}. Default value: `AKMode`. `expandingFactor` ∈ [10, 200]. Default value: 150. |

### `AscendIndexSearchParams`<a name="ZH-CN_TOPIC_0000002044950949"></a>

<a name="table414612258177"></a>

| API Definition | `AscendIndexSearchParams(size_t n, std::vector<float>& queryData, int topK, std::vector<float>& dists, std::vector<int64_t>& labels);` |
| --- | --- |
| Description | Search parameter structure for retrieval. |
| `Input` | None |
| Output | None |
| Parameters | `size_t n`: Number of query feature vectors.<br>`std::vector<float>& queryData`: Feature-vector data.<br>`int topK`: Number of most similar results to return.<br>`std::vector<float>& dists`: Distance values between the query vectors and the top `topK` nearest vectors.<br>`std::vector<int64_t>& labels`: IDs of the top `topK` nearest vectors to the query. When the number of valid search results is less than `topK`, the remaining invalid labels are filled with `-1`. |
| Constraints | `topK` ∈ (0, 4096]. `n` ∈ (0, 10000]. `queryData` cannot be empty, and its data length must be greater than or equal to `n * dim`. `dists` cannot be empty, and the length of the data it points to must be greater than or equal to `n * topK`. `labels` cannot be empty, and the length of the data it points to must be greater than or equal to `n * topK`. |

## `AscendIndexIVFFlat`<a name="ZH-CN_TOPIC_0000002478095516"></a>

### Overview<a name="ZH-CN_TOPIC_0000002510095475"></a>

`AscendIndexIVFFlat` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only IP distance.

### `AscendIndexIVFFlat`<a name="ZH-CN_TOPIC_0000002509975505"></a>

| API Definition | `AscendIndexIVFFlat(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFFlatConfig config)` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFFlat`, which creates a retrieval `Index` on Ascend. |
| Input | `int dims`: Dimensionality of the base-index retrieval vectors.<br>`faiss::MetricType metric`: Distance type. Currently only `faiss::METRIC_INNER_PRODUCT` is supported.<br>`int nlist`: Number of IVF buckets.<br>`AscendIndexIVFFlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` currently supports only 128. `nlist` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. |

<a name="table663150151113"></a>

| API Definition | `AscendIndexIVFFlat& operator=(const AscendIndexIVFFlat&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of this index as deleted, making the type non-copyable. |
| Input | `const AscendIndexIVFFlat&`: Constant `AscendIndexIVFFlat`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexIVFFlat`<a name="ZH-CN_TOPIC_0000002477935546"></a>

| API Definition | `~AscendIndexIVFFlat()` |
| --- | --- |
| Description | Destructor of `AscendIndexIVFFlat`, which destroys the `AscendIndexIVFFlat` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `operator=`<a name="ZH-CN_TOPIC_0000002484264062"></a>

| API Definition | `AscendIndexIVFFlat& operator=(const AscendIndexIVFFlat&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of this `Index` as deleted, making the type non-copyable. |
| Input | `const AscendIndexIVFFlat&`: Constant `AscendIndexIVFFlat`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `train`<a name="ZH-CN_TOPIC_0000002478095518"></a>

| API Definition | `void train(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Trains `AscendIndexIVFFlat`, inheriting the relevant APIs from `AscendIndex` and providing a concrete implementation. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const float *x`: Feature-vector data. |
| Output | None |
| Returns | None |
| Constraints | Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of `n` here is `0 < n < 1e9`. The pointer `x` must be non-null, and its length must be `dims * n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Currently only CPU clustering is supported, and `useKmeansPP` cannot be set to `true`. |

## `AscendIndexIVFPQ`<a name="ZH-CN_TOPIC_0000002478095516"></a>

### Overview<a name="ZH-CN_TOPIC_0000002510095475"></a>

`AscendIndexIVFPQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports only L2 distance and, for performance reasons, only retrieval top-k values within 320.

### `AscendIndexIVFPQ`<a name="ZH-CN_TOPIC_0000002509975505"></a>

| API Definition | `AscendIndexIVFPQ(int dims, faiss::MetricType metric, int nlist, int msubs, int nbits, AscendIndexIVFPQConfig config)` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFPQ`, which creates a retrieval `Index` on Ascend. |
| Input | `int dims`: Dimensionality of the base-index retrieval vectors.<br>`faiss::MetricType metric`: Distance type. Currently only `faiss::METRIC_L2` is supported.<br>`int nlist`: Number of IVF buckets.<br>`int msubs`: Number of subspaces to split into.<br>`int nbits`: Number of bits in the PQ code length. For example, when `nbits = 8`, the PQ code indices range from 0 to 255.<br>`AscendIndexIVFPQConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` currently supports only 128. `nlist` ∈ {1024, 2048, 4096, 8192, 16384}. `msubs` ∈ {2, 4, 8, 16}. `nbits` currently supports only 8. `config.useKmeansPP` indicates whether NPU clustering is enabled. Currently only `useKmeansPP = false` is supported, which means CPU clustering only. |

<a name="table663150151113"></a>

| API Definition | `AscendIndexIVFPQ& operator=(const AscendIndexIVFPQ&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of this index as deleted, making the type non-copyable. |
| Input | `const AscendIndexIVFPQ&`: Constant `AscendIndexIVFPQ`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexIVFPQ`<a name="ZH-CN_TOPIC_0000002477935546"></a>

| API Definition | `~AscendIndexIVFPQ()` |
| --- | --- |
| Description | Destructor of `AscendIndexIVFPQ`, which destroys the `AscendIndexIVFPQ` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `operator=`<a name="ZH-CN_TOPIC_0000002484264062"></a>

| API Definition | `AscendIndexIVFPQ& operator=(const AscendIndexIVFPQ&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of this `Index` as deleted, making the type non-copyable. |
| Input | `const AscendIndexIVFPQ&`: Constant `AscendIndexIVFPQ`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `train`<a name="ZH-CN_TOPIC_0000002478095518"></a>

| API Definition | `void train(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Trains `AscendIndexIVFPQ`, inheriting the relevant APIs from `AscendIndex` and providing a concrete implementation. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const float *x`: Feature-vector data. |
| Output | None |
| Returns | None |
| Constraints | Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of `n` here is `0 < n < 1e9`. The pointer `x` must be non-null, and its length must be `dims * n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Setting `useKmeansPP` to `true` enables NPU clustering; otherwise CPU clustering is used. |

### `remove_ids`<a name="ZH-CN_TOPIC_0000002478095518"></a>

| API Definition | `void remove_ids(size_t n, const idx_t *ids);` |
| --- | --- |
| Description | Removes the trained vectors in `AscendIndexIVFPQ` corresponding to the provided index IDs, by calling the relevant APIs in `AscendIndexIVFPQImpl`. |
| Input | `size_t n`: Number of feature vectors to delete.<br>`const idx_t *ids`: IDs of the feature vectors to delete. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `n` here is `0 < n < 1e9`. The pointer `ids` must be non-null, and its length must be `n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. |

### `copyFrom`<a name="ZH-CN_TOPIC_0000002478095518"></a>

| API Definition | `void copyFrom(const faiss::IndexIVFPQ *index);` |
| --- | --- |
| Description | Reads trained data from the `IndexIVFPQ` index. |
| Input | `const faiss::IndexIVFPQ *index`: `IVFPQ` index, a type of index in the Faiss library. |
| Output | None |
| Returns | None |
| Constraints | Before calling this API, ensure that the data in `index` already has trained centroids and an inverted list, and that all parameters are complete. |

### `copyTo`<a name="ZH-CN_TOPIC_0000002478095518"></a>

| API Definition | `void copyTo(const faiss::IndexIVFPQ *index);` |
| --- | --- |
| Description | Saves the trained data into the `IndexIVFPQ` index. |
| Input | `const faiss::IndexIVFPQ *index`: `IVFPQ` index, a type of index in the Faiss library. |
| Output | None |
| Returns | None |
| Constraints | Before calling this API, ensure that the original vectors have been trained and added to the index, so that no empty centroids, codebooks, or inverted lists are read into `index`. |

### `update`<a name="ZH-CN_TOPIC_0000002478095518"></a>

| API Definition | `std::vector<idx_t> update(idx_t n, const float *x, idx_t *ids)` |
| --- | --- |
| Description | Batch-updates the vectors in the `AscendIndexIVFPQ` base index corresponding to `ids` to `x`. IDs that do not exist in the base index are not updated, and the list of missing IDs is returned. |
| Input | `idx_t n`: Number of feature vectors to update.<br>`float *x`: List of feature vectors to update.<br>`idx_t *ids`: List of feature-vector IDs to update. |
| Output | None |
| Returns | `std::vector<idx_t> noExistIds`: Returns the list of vector IDs that do not exist. |
| Constraints | The valid range of `n` here is `0 < n < 1e9`. The pointer `x` must be non-null, and its length must be `dims * n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. The pointer `ids` must be non-null, and its length must be `n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. |

## `AscendIndexIVFRaBitQ`<a name="ZH-CN_TOPIC_0000002513157720"></a>

### Overview<a name="ZH-CN_TOPIC_0000002544797635"></a>

`AscendIndexIVFRaBitQ` uses IVF for acceleration and is a second-level approximate retrieval algorithm. It currently supports L2 distance computation.

### `AscendIndexIVFRaBitQ`<a name="ZH-CN_TOPIC_0000002513317654"></a>

| API Definition | `AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFRaBitQConfig config)` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFRaBitQ`, which creates a retrieval `Index` on Ascend. |
| Input | `int dims`: Dimensionality of the base-index retrieval vectors.<br>`faiss::MetricType metric`: Distance type. Supports `faiss::METRIC_L2` and `faiss::METRIC_IP`.<br>`int nlist`: Number of IVF buckets.<br>`AscendIndexIVFRaBitQConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` currently supports only 128. `nlist` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. |

<a name="table663150151113"></a>

| API Definition | `AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of this index as deleted, making the type non-copyable. |
| Input | `const AscendIndexIVFRaBitQ&`: Constant `AscendIndexIVFRaBitQ`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexIVFRaBitQ`<a name="ZH-CN_TOPIC_0000002544837623"></a>

| API Definition | `~AscendIndexIVFRaBitQ()` |
| --- | --- |
| Description | Destructor of `AscendIndexIVFRaBitQ`, which destroys the `AscendIndexIVFRaBitQ` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `operator=`<a name="ZH-CN_TOPIC_0000002513157724"></a>

| API Definition | `AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of this `Index` as deleted, making the type non-copyable. |
| Input | `const AscendIndexIVFRaBitQ&`: Constant `AscendIndexIVFRaBitQ`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `train`<a name="ZH-CN_TOPIC_0000002544797639"></a>

| API Definition | `void train(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Trains `AscendIndexIVFRaBitQ`, inheriting the relevant APIs from `AscendIndex` and providing a concrete implementation. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const float *x`: Feature-vector data. |
| Output | None |
| Returns | None |
| Constraints | Training uses k-means clustering. A relatively small training set may affect retrieval accuracy. The valid range of `n` here is `0 < n < 1e9`. The pointer `x` must be non-null, and its length must be `dims * n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. Setting `useKmeansPP` to `true` enables NPU clustering; otherwise CPU clustering is used. For precision issues, see floating-point computation precision issues. |

### `remove_ids`<a name="ZH-CN_TOPIC_0000002513157728"></a>

| API Definition | `void remove_ids(size_t n, const idx_t* ids);` |
| --- | --- |
| Description | Removes the trained vectors in `AscendIndexIVFRaBitQ` corresponding to the provided index IDs, by calling the relevant APIs in `AscendIndexIVFRaBitQImpl`. |
| Input | `size_t n`: Number of feature vectors to delete.<br>`const idx_t *ids`: IDs of the feature vectors to delete. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `n` here is `0 < n < 1e9`. The pointer `ids` must be non-null, and its length must be `n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. |

### `copyFrom`<a name="ZH-CN_TOPIC_0000002557609263"></a>

| API Definition | `void copyFrom(const faiss::IndexIVFRaBitQ *index)` |
| --- | --- |
| Description | Provides a CPU-side `IndexIVFRaBitQ` index, loads data from the trained index to the device side for subsequent retrieval, and calls the relevant APIs in `AscendIndexIVFRaBitQImpl`. |
| Input | `const faiss::IndexIVFRaBitQ *index`: Trained CPU-side `IndexIVFRaBitQ` index. |
| Output | None |
| Returns | None |
| Constraints | The pointer `index` must be non-null, and it must point to a trained `IndexIVFRaBitQ` index. Before calling this API to read data, configure `AscendIndexIVFRaBitQConfig` and create an `AscendIndexIVFRaBitQ` object according to the normal procedure. |

### `copyTo`<a name="ZH-CN_TOPIC_0000002557689209"></a>

| API Definition | `void copyTo(faiss::IndexIVFRaBitQ *index) const` |
| --- | --- |
| Description | Provides a CPU-side `IndexIVFRaBitQ` index, downloads the trained data from the device side into the CPU index for persistence, and calls the relevant APIs in `AscendIndexIVFRaBitQImpl`. |
| Input | `const faiss::IndexIVFRaBitQ *index`: Trained CPU-side `IndexIVFRaBitQ` index. |
| Output | None |
| Returns | None |
| Constraints | The pointer `index` must be non-null. Before calling this API to persist data, create an `AscendIndexIVFRaBitQ` object and train it into the index according to the normal procedure. |

### `update`<a name="ZH-CN_TOPIC_0000002566242121"></a>

<a name="table962730101715"></a>

| API Definition | `std::vector<idx_t> update(idx_t n, float* x, idx_t* ids)` |
| --- | --- |
| Description | Batch-updates the vectors in the `AscendIndexIVFRaBitQ` base index corresponding to `ids` to `x`. IDs that do not exist in the base index are not updated, and the list of missing IDs is returned. |
| Input | `idx_t n`: Number of feature vectors to update.<br>`float* x`: List of feature vectors to update.<br>`idx_t *ids`: List of feature-vector IDs to update. |
| Output | None |
| Returns | `std::vector<idx_t> noExistIds`: Returns the list of vector IDs that do not exist. |
| Constraints | The valid range of `n` here is `0 < n < 1e9`. The pointer `x` must be non-null, and its length must be `n * dim`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. The pointer `ids` must be non-null, and its length must be `n`; otherwise, out-of-bounds read/write errors may occur and cause the program to crash. |

## `AscendIndexIVFRaBitQConfig`<a name="ZH-CN_TOPIC_0000002544944511"></a>

`AscendIndexIVFRaBitQ` must use the corresponding `AscendIndexIVFRaBitQConfig` to initialize the relevant resources.

### `Member Overview`<a name="section4211138173219"></a>

<a name="table388535175015"></a>

| Member | Type | Description |
| --- | --- | --- |
| useRandomOrthogonalMatrix | bool | Whether to use a random orthogonal matrix. Default: `true`. |
| needRefine | bool | Whether refinement is required. Default: `false`. |
| matrixSeed | int | Random seed used to generate the random orthogonal matrix. Default: 12345. |
| refineAlpha | float | Refinement-related parameter. During retrieval, if the original plan is to retrieve the top `k`, refinement retrieves the top `k * refineAlpha` results first, and then takes the top `k` from them.<br>The default value is 2. A larger value gives higher recall but lower retrieval efficiency. |

### `AscendIndexIVFRaBitQConfig`<a name="section6579185362314"></a>

>`Note:`
>`AscendIndexIVFRaBitQConfig` inherits from [AscendIndexIVFConfig](./approximate_retrieval.md#ascendindexivfconfig).

| API Definition | `inline AscendIndexIVFRaBitQConfig();` |
| --- | --- |
| Description | Default constructor. The default `devices` is `{0}`. Computation uses the Ascend AI Processor with ID 0, and the default `resource` is `128 MB`. |
| `Input` | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table3725347611"></a>

| API Definition | `inline AscendIndexIVFRaBitQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFRaBitQConfig`, which creates an `AscendIndexIVFRaBitQConfig`. It configures device-side Ascend AI Processor resources according to the values in `devices`, sets the resource pool size, and performs default initialization. |
| `Input` | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is `IVF_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base index size and the `search` batch size. When the base index is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The maximum number is 64. The configured value of `resourceSize` must not exceed `4 * 1024 MB` (`4 * 1024 * 1024 * 1024` bytes). When set to `-1`, the device-side Ascend AI Processor resource is configured to the default value of `128 MB`. |

<a name="table745471811619"></a>

| API Definition | `inline AscendIndexIVFRaBitQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFRaBitQConfig`, which creates an `AscendIndexIVFRaBitQConfig`. It configures device-side Ascend AI Processor resources according to the values in `devices`, sets the resource pool size, and performs default initialization. |
| `Input` | `std::vector<int> devices`: Device-side device IDs.<br>`int resourceSize`: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is `IVF_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base index size and the `search` batch size. When the base index is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The maximum number is 64. The configured value of `resourceSize` must not exceed `4 * 1024 MB` (`4 * 1024 * 1024 * 1024` bytes). When set to `-1`, the device-side Ascend AI Processor resource is configured to the default value of `128 MB`. |

<a name="table1037111614358"></a>

| API Definition | `inline AscendIndexIVFRaBitQConfig(std::vector<int> devices, bool useRandomOrthogonalMatrix_, bool needRefine_, int matrixSeed_, float alpha_, int64_t resourceSize = IVF_DEFAULT_MEM);` |
| --- | --- |
| Description | Constructor for `AscendIndexIVFRaBitQConfig`, which creates an `AscendIndexIVFRaBitQConfig`. It performs initialization according to the input parameters. |
| `Input` | `std::vector<int> devices`: Device-side device IDs.<br>`bool useRandomOrthogonalMatrix_`: Whether to use a random orthogonal matrix.<br>`bool needRefine_`: Whether refinement is required.<br>`int matrixSeed_`: Random seed used to generate the random orthogonal matrix.<br>`float alpha_`: Refinement-related parameter.<br>`int resourceSize`: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation. The default parameter is `IVF_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base index size and the `search` batch size. When the base index is greater than or equal to 10 million and the batch size is greater than or equal to 16, 1024 MB is recommended. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The maximum number is 64. The configured value of `resourceSize` must not exceed `4 * 1024 MB` (`4 * 1024 * 1024 * 1024` bytes). When set to `-1`, the device-side Ascend AI Processor resource is configured to the default value of `128 MB`. |
