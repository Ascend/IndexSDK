# Full Retrieval<a name="ZH-CN_TOPIC_0000001533164645"></a>

## `AscendIndex`<a id="ZH-CN_TOPIC_0000001456375304"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506414937"></a>

AscendIndex is the base class of the `Index` implementations for most retrieval methods in the feature retrieval component. It sits on top of Faiss and defines interfaces for the other indexes in feature retrieval.

### `add`<a id="ZH-CN_TOPIC_0000001506614985"></a>

| API Definition | ```void add(idx_t n, const float *x) override;``` |
| --- | --- |
| Description | Implements AscendIndex index creation and adds new feature vectors to the base library. When you add features with `add`, the default IDs of the corresponding features are [0, `ntotal`). |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const float *x`: Feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.<br>`n` must be in the range `0 < n < 1e9`.<br>Note:<br>The `add` interface cannot be used together with the `add_with_ids` interface. After you use the `add` interface, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` interface. |

<a name="table17254342193617"></a>

| API Definition | `void add(idx_t n, const uint16_t *x);` |
| --- | --- |
| Description | Implements AscendIndex index creation and adds new feature vectors to the base library. When you add features with `add`, the default IDs of the corresponding features are [0, `ntotal`). |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const uint16_t *x`: Feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.<br>`n` must be in the range `0 < n < 1e9`. |

### `add_with_ids`<a id="ZH-CN_TOPIC_0000001456694864"></a>

| API Definition | `void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;` |
| --- | --- |
| Description | Implements AscendIndex index creation and adds new feature vectors to the base library, with an ID for each base-library feature. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const float *x`: Feature vectors to add to the base library.<br>`const idx_t *ids`: IDs of the feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`, and the length of pointer `ids` must be `n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `n` must be in the range `0 < n < 1e9`. When the `filterable` filter switch is set to `true`, ensure that the timestamps in `ids` are positive.<br>`ids` of type `uint64_t` contain `timestamp` of type `int32_t` and `cid` of type `uint8_t`, as shown below:<br> -----&#124; cid &#124; timestamp &#124; ----- <br>14 &#124; 8 &#124; 32 &#124; 10 |

<a name="table562574920111"></a>

| API Definition | `void add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids);` |
| --- | --- |
| Description | Implements AscendIndex index creation and adds new feature vectors to the base library, with an ID for each base-library feature. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const uint16_t *x`: Feature vectors to add to the base library.<br>`const idx_t *ids`: IDs of the feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`, and the length of pointer `ids` must be `n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `n` must be in the range `0 < n < 1e9`. When the `filterable` filter switch is set to `true`, ensure that the timestamps in `ids` are positive. `ids` of type `uint64_t` contain `timestamp` of type `int32_t` and `cid` of type `uint8_t`, as shown below: <br>-----&#124; cid &#124; timestamp &#124; ----- <br>14  &#124;  8  &#124; 32  &#124;  10 |

### `AscendIndex`<a name="ZH-CN_TOPIC_0000001456695048"></a>

| API Definition | `AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config)` |
| --- | --- |
| Description | Constructor of `AscendIndex`. It creates an `AscendIndex` with dimension `dims`. A single `Index` manages vectors with one fixed dimension. Device-side resources are set according to the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndex`.<br>`faiss::MetricType metric`: Distance metric used by `AscendIndex` when performing feature-vector similarity retrieval. Currently supported values are `faiss::MetricType::METRIC_L2` and `faiss::MetricType::METRIC_INNER_PRODUCT`.<br>`AscendIndexConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` must be an integer in the range (0, 4096] and must be divisible by 16. |

<a name="table161511529133912"></a>

| API Definition | `AscendIndex(const AscendIndex&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor of `AscendIndex` as deleted. Therefore, `AscendIndex` is a non-copyable type. |
| Input | `const AscendIndex&`: Constant `AscendIndex`. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table62621513124018"></a>

| API Definition | `virtual ~AscendIndex();` |
| --- | --- |
| Description | Destructor of `AscendIndex`. It destroys the `AscendIndex` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `getDeviceList`<a name="ZH-CN_TOPIC_0000001506495857"></a>

| API Definition | `std::vector<int> getDeviceList();` |
| --- | --- |
| Description | Returns the device-side Ascend AI Processor configuration managed in `Index`. Derived classes provide the implementation. This class does not provide one and returns only an empty `vector<int>`. |
| Input | None |
| Output | None |
| Returns | Device-side Ascend AI Processor configuration managed in `Index`. |
| Constraints | None |

### `operator=`<a name="ZH-CN_TOPIC_0000001506334661"></a>

| API Definition | `AscendIndex& operator=(const AscendIndex&) = delete;` |
| --- | --- |
| Description | Declares the copy assignment operator of `AscendIndex` as deleted. Therefore, `AscendIndex` is a non-copyable type. |
| Input | `const AscendIndex&`: Constant `AscendIndex`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `reclaimMemory`<a name="ZH-CN_TOPIC_0000001456695092"></a>

| API Definition | `virtual size_t reclaimMemory();` |
| --- | --- |
| Description | Reduces the memory occupied by the base library without changing the number of vectors in it. The implementation is inherited and provided by derived classes. This class does not provide an implementation. |
| Input | None |
| Output | None |
| Returns | Size of the reclaimed memory, in bytes. |
| Constraints | None |

### `remove_ids`<a name="ZH-CN_TOPIC_0000001456535000"></a>

| API Definition | `size_t remove_ids(const faiss::IDSelector &sel) override;` |
| --- | --- |
| Description | Removes the specified feature vectors from the base library in `AscendIndex`. |
| Input | `const faiss::IDSelector &sel`: Feature vectors to be deleted. For details about usage and definition, see the corresponding Faiss documentation. |
| Output | None |
| Returns | Number of deleted feature vectors. |
| Constraints | None |

### `reserveMemory`<a name="ZH-CN_TOPIC_0000001456375348"></a>

| API Definition | `virtual void reserveMemory(size_t numVecs);` |
| --- | --- |
| Description | Abstract interface for reserving memory for the base library before it is built. The implementation is inherited and provided by derived classes. This class does not provide an implementation. |
| Input | `size_t numVecs`: Number of vectors in the base library for which to reserve memory. |
| Output | None |
| Returns | None |
| Constraints | None |

### `reset`<a name="ZH-CN_TOPIC_0000001506414901"></a>

| API Definition | `void reset() override;` |
| --- | --- |
| Description | Clears the base-library vectors of this `AscendIndex`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `search`<a name="ZH-CN_TOPIC_0000001506334641"></a>

| API Definition | `void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;` |
| --- | --- |
| Description | Feature-vector retrieval interface. It returns the IDs of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n`: Number of query feature vectors.<br>`const float *x`: Feature-vector data.<br>`idx_t k`: Number of most similar results to return.<br>`const SearchParameters *params`: Optional parameter of Faiss. The default value is `nullptr`. This parameter is not supported currently. |
| Output | `float *distances`: Distance values between the query vectors and the top `k` nearest vectors. When the number of valid retrieval results is fewer than `k`, fill the remaining invalid distances with 65504 or -65504, depending on the metric.<br>`idx_t *labels`: IDs of the top `k` nearest vectors to the query. When the number of valid retrieval results is fewer than `k`, fill the remaining invalid labels with -1. |
| Returns | None |
| Constraints | The length of query feature-vector data `x` must be `dims * n`, and the lengths of `distances` and `labels` must be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `n` must be in the range `0 < n < 1e9`. `k` is usually not allowed to exceed 4096. |
| Note | In scenarios that use the small-base-library brute-force algorithm, if performance drops when the base library and batch size are large, increase the `resources` parameter in `AscendIndexConfig`. The default value of the brute-force algorithm is 128 MB. |

<a name="table03178548130"></a>

| API Definition | void search(idx_t n, const uint16_t *x, idx_t k, float*distances, idx_t *labels) const; |
| --- | --- |
| Description | Feature-vector retrieval interface of `AscendIndex`. It returns the IDs of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n`: Number of query feature vectors.<br>`const uint16_t *x`: Feature-vector data.<br>`idx_t k`: Number of most similar results to return. |
| Output | `float *distances`: Distance values between the query vectors and the top `k` nearest vectors. When the number of valid retrieval results is fewer than `k`, fill the remaining invalid distances with 65504 or -65504, depending on the metric.<br>`idx_t *labels`: IDs of the top `k` nearest vectors to the query. When the number of valid retrieval results is fewer than `k`, fill the remaining invalid labels with -1. |
| Returns | None |
| Constraints | The length of query feature-vector data `x` must be `dims * n`, and the lengths of `distances` and `labels` must be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `n` must be in the range `0 < n < 1e9`. `k` is usually not allowed to exceed 4096. |
| Note | In scenarios that use the small-base-library brute-force algorithm, if performance drops when the base library and batch size are large, increase the `resources` parameter in `AscendIndexConfig`. The default value of the brute-force algorithm is 128 MB. |

## `AscendIndexCluster`<a id="ZH-CN_TOPIC_0000001614744825"></a>

### Overview<a name="ZH-CN_TOPIC_0000001564586790"></a>

`AscendIndexCluster` requires [`Init`](#init) to initialize the specified resources. After initialization, it allocates a complete memory space to store the base library. After use, call [`Finalize`](#finalize) to release the resources.

`AscendIndexCluster` supports only the vector inner-product distance type in standard mode on Atlas Inference Series products. It depends on Flat and AICPU operators. For details, see [Flat](../user_guide.md#generating-operators) and [AICPU](../user_guide.md#generating-operators).

It supports multithreaded concurrent calls. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `AddFeatures`<a name="ZH-CN_TOPIC_0000001614746533"></a>

| API Definition | `APP_ERROR AddFeatures(int n, const float *features, const uint32_t *indices);` |
| --- | --- |
| Description | Inserts `n` feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, this interface updates it. |
| Input | `int n`: Number of feature vectors to insert.<br>`const float *features`: Feature vectors to insert. The length is `n` multiplied by the vector dimension `dim`.<br>`const uint32_t *indices`: Indices of the feature vectors to insert. The valid length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `indices`: The index of each feature must be in [0, `capacity` ), and `indices` must be continuous. `n`: Must be in (0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

<a name="table772538154310"></a>

| API Definition | `APP_ERROR AddFeatures(int n, const uint16_t *features, const int64_t *indices);` |
| --- | --- |
| Description | Inserts `n` feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, this interface updates it. |
| Input | `int n`: Number of feature vectors to insert.<br>`const uint16_t *features`: Feature vectors to insert. The length is `n` multiplied by the vector dimension `dim`.<br>`const int64_t *indices`: Indices of the feature vectors to insert. The valid length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `indices`: The index of each feature must be in [0, `capacity` ). `n`: Must be in (0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `AscendIndexCluster`<a name="ZH-CN_TOPIC_0000001564746410"></a>

| API Definition | `AscendIndexCluster();` |
| --- | --- |
| Description | Constructor of `AscendIndexCluster`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table15621560282"></a>

| API Definition | `AscendIndexCluster(const AscendIndexCluster&) = delete;` |
| --- | --- |
| Description | Declares this `Index` copy constructor as deleted. Therefore, the type is non-copyable. |
| Input | `const AscendIndexCluster&`: `AscendIndexCluster` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexCluster`<a name="ZH-CN_TOPIC_0000002399598393"></a>

<a name="table179216322487"></a>

| API Definition | `virtual ~AscendIndexCluster() = default;` |
| --- | --- |
| Description | Destructor of `AscendIndexCluster`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `ComputeDistanceByIdx`<a name="ZH-CN_TOPIC_0000002446061685"></a>

| API Definition | `APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | `ComputeDistance` calculates the distance between the query vectors and all base-library vectors, whereas `ComputeDistanceByIdx` calculates only the distance between the query vectors and the base-library vectors at the given indices. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the interface returns the mapped top-`k` results. |
| Input | `int n`: Number of query feature vectors.<br>`const uint16_t *queries`: Query feature vectors. The valid length is `n * dim`, and `dim` must be the same as the dimension specified during initialization.<br>`const int *num`: Number of base-library feature vectors to compare for each query. The length is `n`.<br>`const uint32_t *indices`: Indices of the base-library feature vectors to compare. The number of base-library vectors to compare can differ for each query. Valid vector indices must be stored continuously from front to back, and the space usage must be padded according to the maximum `num`. The length of `indices` is `n * max(num)`.<br>`unsigned int tableLen`: Mapping-table length. The default value is `0`, which means that no mapping is performed. Currently, the supported mapping-table length is `10000`.<br>`const float *table`: Mapping-table pointer that points to valid mapped values of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `*table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: Distances between the query vectors and the selected base-library vectors. For each query, valid distances are recorded continuously from front to back, and the space usage is padded according to the maximum `num`. The total length is `n * max(num)`. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `n`: Must be in the range (0, `capacity` ]. `num`: User-specified. The length is `n`, and the `num` value for each query must be in [0, `ntotal`]. `indices`: The index of each feature must be in [0, `ntotal` ). Example parameter values: `n = 3`, `num[3] = {1, 3, 5}` means that the three queries compare against 1, 3, and 5 base-library vectors respectively. If `max(num) = 5`, then the space pointed to by `indices` is aligned to 5, and the total size is `3 * 5 * sizeof(idx_t)` bytes, for example `{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}`. When both `tableLen` and `table` meet the requirements, the interface maps the computed `distance` values.<br>First, normalize `distance` to a floating-point value `f1` in [0, 1]. Then multiply `f1` by `tableLen` and round it down to obtain an integer index in [0, `tableLen`]. Next, use the integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`. This completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be abstracted as `((CosDistance + 1) / 2) * tableLen`. |

### `ComputeDistanceByThreshold`<a name="ZH-CN_TOPIC_0000001615066169"></a>

> This interface must be used together with [`AddFeatures(int n, const float *features, const uint32_t *indices);`](#addfeatures).

| API Definition | `APP_ERROR ComputeDistanceByThreshold(const std::vector<uint32_t> &queryIdxArr, uint32_t codeStartIdx, uint32_t codeNum, float threshold, bool aboveFilter, std::vector<std::vector<float>> &resDistArr, std::vector<std::vector<uint32_t>> &resIdxArr);` |
| --- | --- |
| Description | Calculates the distances between the queried feature vectors in the base library and the specified base-library feature vectors, then filters by threshold and returns the distances and labels that meet the conditions. |
| Input | `const std::vector<uint32_t> &queryIdxArr`: Indices of the vectors to query in the base library.<br>`uint32_t codeStartIdx`: Starting index of the base library vectors for distance calculation.<br>`uint32_t codeNum`: Number of base-library vectors for distance calculation.<br>`float threshold`: Threshold used for filtering. Distances smaller than the threshold are filtered out.<br>`bool aboveFilter`: Reserved parameter. |
| Output | `std::vector<std::vector<float>> &resDistArr`: Two-dimensional array that returns the distances between each query vector and the base-library vectors that meet the threshold condition.<br>`std::vector<std::vector<uint32_t>> &resIdxArr`: Two-dimensional array that returns the indices of the base-library vectors that meet the threshold condition for each query vector. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | The lengths of `queryIdxArr`, `resDistArr`, and `resIdxArr` must be the same, that is, `queryIdxArr.size() == resDistArr.size()`. `queryIdxArr.size()` must be greater than `0` and less than or equal to `ntotal`. `codeNum` must be greater than `0` and less than or equal to `ntotal`. `codeStartIdx + codeNum` must not exceed `ntotal` (the base-library size). `codeStartIdx` must be greater than or equal to `0` and less than or equal to `ntotal`. |

### `Finalize`<a name="ZH-CN_TOPIC_0000001614906601"></a>

| API Definition | void Finalize(); |
| --- | --- |
| Description | Releases feature-library management resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `GetFeatures`<a name="ZH-CN_TOPIC_0000002412742482"></a>

| API Definition | `APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices);` |
| --- | --- |
| Description | Retrieves `n` feature vectors at the specified indices. |
| Input | `int n`: Number of base-library vectors to retrieve.<br>`const int64_t *indices`: Indices corresponding to the feature vectors. The length is `n`. |
| Output | `uint16_t *features`: Feature vectors corresponding to the queried indices. The length is `n * vector dimension dim`. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `indices`: The index of each feature must be in [0, `ntotal` ), and `ntotal` can be obtained through the `GetNTotal` interface. `n`: Must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetNTotal`<a name="ZH-CN_TOPIC_0000002412582646"></a>

| API Definition | `int GetNTotal() const;` |
| --- | --- |
| Description | Queries the theoretical maximum number of feature vectors in the current feature library. If the inserted feature-vector indices are continuous, `ntotal` is equal to the number of feature vectors. |
| Input | None |
| Output | `int ntotal`: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1. |
| Returns | `int`: Theoretical maximum number of feature vectors, which is the maximum base-library vector index plus 1. |
| Constraints | None |

### `Init`<a name="ZH-CN_TOPIC_0000001614866169"></a>

| API Definition | `APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector<int> &deviceList, int64_t resourceSize = -1);` |
| --- | --- |
| Description | Initialization function of `AscendIndexCluster`. |
| Input | `int dim`: Dimension of the feature vectors managed by `AscendIndexCluster`.<br>`int capacity`: Maximum base-library capacity. The interface allocates `capacity * dim * sizeof(fp16)` bytes of memory based on the value of `capacity`.<br>`faiss::MetricType metricType`: Feature-distance category, including vector inner product, Euclidean distance, and cosine similarity.<br>`const std::vector<int> &deviceList`: Device-side resource configuration.<br>`int64_t resourceSize`: Size of the preallocated memory pool on the device side, in bytes. This memory stores intermediate results during computation and is used to avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `-1`, which means `128 MB`. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `dim` must be one of `{32, 64, 128, 256, 384, 512}`. `metricType`: `AscendIndexCluster` currently implements only vector inner-product distance, which means that only `faiss::MetricType::METRIC_INNER_PRODUCT` is supported. The maximum memory that can be allocated for the base library is `12,288,000,000` bytes, and the value range of `capacity` is [0, 12000000]. For example, for a base-library vector with 512 dimensions and the FP16 type, the maximum supported `capacity` is 12 million (`12288000000 / (512 * sizeof(fp_16))`). For base-library vectors with 256 dimensions and the FP16 type, even though the memory constraint supports a larger `capacity`, the maximum `capacity` can still be only 12 million. Only single-card configuration is supported. Multi-card configuration is not supported yet, so `deviceList.size()` must equal `1`. `resourceSize` can be `-1` or a value in [134217728, 4294967296], which is equivalent to `[128 MB, 4096 MB]`. This parameter is determined jointly by the base-library size and the `search` batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to `1024 MB`. |

### `operator =`<a name="ZH-CN_TOPIC_0000001897100377"></a>

| API Definition | `AscendIndexCluster& operator=(const AscendIndexCluster&) = delete;` |
| --- | --- |
| Description | Declares this `Index` copy assignment operator as deleted, making the type non-copyable. |
| Input | `const AscendIndexCluster&`: `AscendIndexCluster` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `RemoveFeatures`<a name="ZH-CN_TOPIC_0000002446181741"></a>

| API Definition | `APP_ERROR RemoveFeatures(int n, const int64_t *indices);` |
| --- | --- |
| Description | Removes `n` feature vectors at the specified indices from the vector library. |
| Input | `int n`: Number of feature vectors to remove.<br>`const int64_t *indices`: Indices corresponding to the feature vectors. The length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `indices`: The index of each feature must be in [0, `ntotal` ), and `ntotal` can be obtained through the `GetNTotal` interface. `n`: Must be in [0, `capacity` ]. `indices` must be a non-null pointer, and its length must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `SearchByThreshold`<a name="ZH-CN_TOPIC_0000002446061689"></a>

| API Definition | `APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num, int64_t * indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Adds threshold filtering on top of `Search` and returns only the results that meet the threshold condition. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the interface returns the mapped top-`k` results. |
| Input | `int n`: Number of feature vectors to query.<br>`const uint16_t *queries`: Query feature vectors. The length is `n * dim`.<br>`float threshold`: Threshold used for filtering. The interface does not restrict the value range. If you pass a mapping table, the interface first maps the distance to a score and then filters by `threshold`.<br>`int topk`: Sorts the comparison distances between the query and the base library, then returns the top `k` results.<br>`unsigned int tableLen`: Mapping-table length. The default value is `0`, which means that no mapping is performed. Currently, the supported mapping-table length is `10000`.<br>`const float *table`: Mapping-table pointer that points to valid mapped values of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `*table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `int *num`: Number of base-library vectors that meet the threshold condition for each query feature vector. The length is `n`.<br>`int64_t *indices`: Indices of base-library vectors that meet the threshold condition. For each query, matching indices are recorded from front to back and the space is padded according to `topk`. The total length of `indices` is `n * topk`.<br>`float *distances`: Distances between the base-library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of `indices`. |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `n`: Must be in the range (0, `capacity` ]. `topk`: `k` must be in (0, 1024]. When both `tableLen` and `table` meet the requirements, the interface maps the computed `distance` values.<br>First, normalize `distance` to a floating-point value `f1` in [0, 1]. Then multiply `f1` by `tableLen` and round it down to obtain an integer index in [0, `tableLen`]. Next, use the integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`. This completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be abstracted as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `SetNTotal`<a name="ZH-CN_TOPIC_0000002412742486"></a>

| API Definition | `APP_ERROR SetNTotal(int n);` |
| --- | --- |
| Description | Provides an external way to adjust the `ntotal` count.<br>After base-library vectors are added, the `Index` internally updates `ntotal` according to the maximum inserted index. However, it does not record which areas in the range [0, `ntotal` ] are invalid space. Therefore, the `RemoveFeatures` operation does not change the value of `ntotal`. If you explicitly record the maximum base-library index after add and remove operations in the service layer, you can set `ntotal` manually. This can reduce the amount of work performed by the operators within a controllable range and improve interface performance.<br>For example, if you currently insert 100 vectors with base-library indices from 0 to 99, then `ntotal = 100`. If you delete the base-library vectors with indices from 80 to 90, the internal `ntotal` of `Index` remains unchanged and can only be set to a value in [ `ntotal`, `capacity` ]. If you then delete the base-library vectors with indices from 90 to 99, you can manually set `ntotal` to a value in [80, `capacity` ]. When you set it to `80`, the amount of base-library data participating in the comparison is effectively reduced by 20 vectors. |
| Input | `int n`: Maximum base-library index plus 1, managed by the user in the service layer. |
| Output | None |
| Returns | `APP_ERROR`: Return status of the call. For details, see the API return value reference. |
| Constraints | `n`: Must be in the range [0, `capacity` ]. |

## `AscendIndexConfig`<a name="ZH-CN_TOPIC_0000001506414705"></a>

`AscendIndex` must use the corresponding `AscendIndexConfig` to initialize the relevant resources. `AscendIndexConfig` must configure the hardware resources and memory pool size used during retrieval.

> [!NOTE]
> The memory pool size unit is `Byte`. This parameter specifies the size of the preallocated memory pool on the device side. The memory pool stores the results of distance calculations on Ascend hardware. When the base library is large, you are advised to reserve a larger memory pool.

**Members<a name="section1372191465013"></a>**

|Member|Type|Description|
|--|--|--|
|deviceList|std::vector\<int>|Device-side device IDs.|
|resourceSize|int64_t|Device-side memory pool size, in bytes. The default parameter is `INDEX_DEFAULT_MEM` in the header file.|
|slim|bool|Member variable of `AscendIndexConfig`. Indicates whether to increase memory dynamically.|
|filterable|bool|Member variable of `AscendIndexConfig`. Indicates whether to filter by ID.|
|dBlockSize|uint32_t|Device-side block size configuration.|

**API Description<a name="section1197816229504"></a>**

| API Definition | `AscendIndexConfig()` |
| --- | --- |
| Description | Default constructor of `AscendIndexConfig`. The default `deviceList` is `0`, which means that the Ascend AI Processor with ID `0` on the NPU is used as the heterogeneous computing platform for `AscendFaiss` retrieval. The default resource-pool size is `32 MB` (`32*1024*1024` bytes). |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table0786126165110"></a>

| API Definition | `AscendIndexConfig(std::initializer_list<int> devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)` |
| --- | --- |
| Description | Constructor of `AscendIndexConfig`. It creates an `AscendIndexConfig` and sets device-side Ascend AI Processor resources according to the values configured in `devices`, while also configuring the resource-pool size. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resources`: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is `INDEX_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base library size and the `search` batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to `1024 MB`.<br>`uint32_t blockSize`: Device-side block size configuration. It constrains the amount of data processed in one `tik` operator call and the size of vectors stored in each partition of the base-library shard. The default value of `DEFAULT_BLOCK_SIZE` is `16384 * 16 = 262144`. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The maximum number is 64. The configured value of `resources` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). |

<a name="table23967285518"></a>

| API Definition | `AscendIndexConfig(std::vector<int> devices, int64_t resources = INDEX_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)` |
| --- | --- |
| Description | Constructor of `AscendIndexConfig`. It creates an `AscendIndexConfig` and sets device-side Ascend AI Processor resources according to the values configured in `devices`, while also configuring the resource-pool size. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resources`: Size of the preallocated memory pool on the device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default parameter is `INDEX_DEFAULT_MEM` in the header file. This parameter is determined jointly by the base library size and the `search` batch size. When the base library is greater than or equal to 10 million and the batch size is greater than or equal to 16, you are advised to set it to `1024 MB`.<br>`uint32_t blockSize`: Device-side block size configuration. It constrains the amount of data processed in one `tik` operator call and the size of vectors stored in each partition of the base-library shard. The default value of `DEFAULT_BLOCK_SIZE` is `16384 * 16 = 262144`. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs. The maximum number is 64. The configured value of `resources` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). |

## `AscendIndexInt8`<a id="ZH-CN_TOPIC_0000001506495841"></a>

### Overview<a id="ZH-CN_TOPIC_0000001506495913"></a>

`AscendIndexInt8` is the base class of the indexes that use INT8 feature vectors in the feature retrieval component. It defines interfaces for other INT8 indexes in feature retrieval.

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, users must lock before use, or the retrieval interface may raise exceptions. It also does not support sharing one device across different threads. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `add`<a name="ZH-CN_TOPIC_0000001506334825"></a>

| API Definition | `void add(idx_t n, const int8_t *x);` |
| --- | --- |
| Description | Adds new feature vectors to the `AscendIndexInt8` base library. When you add features with `add`, the default IDs of the corresponding features are [0, `ntotal`). |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const int8_t *x`: Feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is `0 < n < 1e9`. |

<a name="table6211414109"></a>

| API Definition | `void add(idx_t n, const char *x);` |
| --- | --- |
| Description | Adds new feature vectors to the `AscendIndexInt8` base library. When you add features with `add`, the default IDs of the corresponding features are [0, `ntotal`). |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const char *x`: Feature vectors to add to the base library. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is `0 < n < 1e9`. |

> [!NOTE]
>
>- The `add` interface cannot be used together with the `add_with_ids` interface.
>- After you use the `add` interface, the `labels` in the search results may repeat. If your service has requirements for labels, you are advised to use the `add_with_ids` interface.

### `add_with_ids`<a name="ZH-CN_TOPIC_0000001506614905"></a>

| API Definition | `void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);` |
| --- | --- |
| Description | Adds new feature vectors to the `AscendIndexInt8` base library and specifies the feature IDs. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const int8_t *x`: Feature vectors to add to the base library.<br>`const idx_t *ids`: IDs of the feature vectors to add to the base library. The IDs must be unique within the `Index` instance. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`, and the length of pointer `ids` must be `n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is `0 < n < 1e9`. |

<a name="table38814511704"></a>

| API Definition | `void add_with_ids(idx_t n, const char *x, const idx_t *ids);` |
| --- | --- |
| Description | Adds new feature vectors to the `AscendIndexInt8` base library and specifies the feature IDs. |
| Input | `idx_t n`: Number of feature vectors to add to the base library.<br>`const char *x`: Feature vectors to add to the base library.<br>`const idx_t *ids`: IDs corresponding to the feature vectors to add to the base library. The IDs must be unique within the `Index` instance. |
| Output | None |
| Returns | None |
| Constraints | The length of pointer `x` must be `dims * n`, and the length of pointer `ids` must be `n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The valid range of the total number of base-library vectors is `0 < n < 1e9`. |

### `assign`<a name="ZH-CN_TOPIC_0000001506495721"></a>

| API Definition | `void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);` |
| --- | --- |
| Description | Feature-vector retrieval interface of `AscendIndexInt8`. It returns the IDs of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n`: Number of query feature vectors.<br>`const int8_t *x`: Feature-vector data.<br>`idx_t k`: Number of most similar results to return. |
| Output | `idx_t *labels`: IDs of the top `k` nearest vectors to the query. |
| Returns | None. |
| Constraints | The length of query feature-vector data `x` must be `dims * n`, and the length of `labels` must be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `n` must be greater than `0` and less than `1e9`. `k` must be greater than `0` and less than or equal to `4096`. `n * k` must be less than `1e10`. |

### `AscendIndexInt8`<a name="ZH-CN_TOPIC_0000001506614993"></a>

| API Definition | `AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config)`; |
| --- | --- |
| Description | Constructor of `AscendIndexInt8`. It creates an `AscendIndexInt8` with dimension `dims`. The dimension of the vector set managed by a single `Index` is unique. Device-side resources are set according to the values configured in `config`. |
| Input | `int dims`: Dimension of a set of feature vectors managed by `AscendIndexInt8`.<br>`faiss::MetricType metric`: Distance metric used by `AscendIndexInt8` when performing feature-vector similarity retrieval. Currently supported values are `faiss::MetricType::METRIC_L2` and `faiss::MetricType::METRIC_INNER_PRODUCT`.<br>`AscendIndexInt8Config config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` must be an integer that is not smaller than 64 and not larger than 1024, and it must be divisible by 64. |

<a name="table103312407520"></a>

| API Definition | `AscendIndexInt8(const AscendIndexInt8&) = delete;` |
| --- | --- |
| Description | Declares this `Index` copy constructor as deleted. Therefore, the type is non-copyable. |
| Input | `const AscendIndexInt8&`: `AscendIndexInt8` object. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table1882220715614"></a>

| API Definition | `virtual ~AscendIndexInt8();` |
| --- | --- |
| Description | Destructor of `AscendIndexInt8`. It destroys the `AscendIndexInt8` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `getDeviceList`<a name="ZH-CN_TOPIC_0000001672982421"></a>

| API Definition | `std::vector<int> getDeviceList() const;` |
| --- | --- |
| Description | Return the device-side Ascend AI Processor settings managed by `Index`. Subclasses inherit from it and implement it. This base class does not provide a corresponding implementation and returns only an empty `vector<int>`. |
| Input | None |
| Output | None |
| Returns | The device-side Ascend AI Processor settings managed by `Index`. |
| Constraints | None |

### `getDim`<a name="ZH-CN_TOPIC_0000001690599922"></a>

| API Definition | `int getDim() const;` |
| --- | --- |
| Description | Get the dimension of the feature vector set managed by `AscendIndexInt8`. |
| Input | None |
| Output | None |
| Returns | The dimension of the feature vector set managed by `AscendIndexInt8`. |
| Constraints | None |

### `getNTotal`<a name="ZH-CN_TOPIC_0000001738718517"></a>

| API Definition | `faiss::idx_t getNTotal() const;` |
| --- | --- |
| Description | Get the number of feature vectors that `AscendIndexInt8` has added to the base vector set. |
| Input | None |
| Output | None |
| Returns | The number of feature vectors that `AscendIndexInt8` has added to the base vector set. |
| Constraints | None |

### `getMetricType`<a name="ZH-CN_TOPIC_0000001738678653"></a>

| API Definition | `faiss::MetricType getMetricType() const;` |
| --- | --- |
| Description | Get the distance metric type used by `AscendIndexInt8` when performing feature vector similarity retrieval. |
| Input | None |
| Output | None |
| Returns | The distance metric type used by `AscendIndexInt8` when performing feature vector similarity retrieval. |
| Constraints | None |

### `isTrained`<a name="ZH-CN_TOPIC_0000001690759666"></a>

| API Definition | `bool isTrained() const;` |
| --- | --- |
| Description | Determine whether `AscendIndexInt8` is trained. |
| Input | None |
| Output | None |
| Returns | The trained state of `AscendIndexInt8`. `true` means trained, and `false` means not trained. |
| Constraints | None |

### `operator =`<a name="ZH-CN_TOPIC_0000001506414841"></a>

| API Definition | `AscendIndexInt8& operator=(const AscendIndexInt8&) = delete;` |
| --- | --- |
| Description | Declare this `Index` assignment operator as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexInt8&`: A constant `AscendIndexInt8`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `reclaimMemory`<a name="ZH-CN_TOPIC_0000001506615133"></a>

| API Definition | `virtual size_t reclaimMemory();` |
| --- | --- |
| Description | A virtual function defined in the base class. See the subclass for details. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `remove_ids`<a name="ZH-CN_TOPIC_0000001456695088"></a>

| API Definition | `size_t remove_ids(const faiss::IDSelector &sel);` |
| --- | --- |
| Description | Implement the interface for deleting the specified feature vectors from the base vector set in `AscendIndexInt8`. |
| Input | `const faiss::IDSelector &sel`: Feature vectors to delete. For details on usage and definition, see the corresponding Faiss documentation. |
| Output | None |
| Returns | The number of deleted feature vectors. |
| Constraints | None |

### `reserveMemory`<a name="ZH-CN_TOPIC_0000001506615065"></a>

| API Definition | `virtual void reserveMemory(size_t numVecs);` |
| --- | --- |
| Description | A virtual function defined in the base class. See the subclass for details. |
| Input | `size_t numVecs`: Number of base vectors for which to reserve memory. |
| Output | None |
| Returns | None |
| Constraints | None |

### `search`<a name="ZH-CN_TOPIC_0000001506414889"></a>

| API Definition | `void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;` |
| --- | --- |
| Description | Implement the feature vector search interface for `AscendIndexInt8`, and return the distances and IDs of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n`: Number of query feature vectors.<br>`const int8_t *x`: Feature vector data.<br>`idx_t k`: Number of most similar results to return. |
| Output | `float *distances`: Distance values between the query vectors and the top `k` nearest vectors. When fewer than `k` valid retrieval results are available, fill the remaining invalid distances with `65504` or `-65504` depending on the metric.<br>`idx_t *labels`: IDs of the top `k` nearest vectors to the query. When fewer than `k` valid retrieval results are available, fill the remaining invalid labels with `-1`. |
| Returns | None. |
| Constraints | The length of the query feature vector data `x` should be `dims * n`, and the lengths of `distances` and `labels` should be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, `n` is greater than `0` and less than `1e9`. Here, `k` is greater than `0` and less than or equal to `4096`. |

<a name="table88671631181418"></a>

| API Definition | `void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;` |
| --- | --- |
| Description | Implement the feature vector search interface for `AscendIndexInt8`, and return the distances and IDs of the `k` most similar features based on the input feature vectors. |
| Input | `idx_t n`: Number of query feature vectors.<br>`const char *x`: Feature vector data.<br>`idx_t k`: Number of most similar results to return. |
| Output | `float *distances`: Distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: IDs of the top `k` nearest vectors to the query. |
| Returns | None. |
| Constraints | The length of the query feature vector data `x` should be `dims * n`, and the lengths of `distances` and `labels` should be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, `n` is greater than `0` and less than `1e9`. Here, `k` is greater than `0` and less than or equal to `4096`. |

### `train`<a name="ZH-CN_TOPIC_0000001456534956"></a>

| API Definition | `virtual void train(idx_t n, const int8_t *x);` |
| --- | --- |
| Description | A virtual function defined in the base class. See the subclass for details. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const int8_t *x`: Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | None |

### `updateCentroids`<a name="ZH-CN_TOPIC_0000001506414833"></a>

| API Definition | `virtual void updateCentroids(idx_t n, const int8_t *x);` |
| --- | --- |
| Description | A virtual function defined in the base class. See the subclass for details. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const int8_t *x`: Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table2023134918146"></a>

| API Definition | `virtual void updateCentroids(idx_t n, const char *x);` |
| --- | --- |
| Description | A virtual function defined in the base class. See the subclass for details. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const char *x`: Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendIndexInt8Config`<a id="ZH-CN_TOPIC_0000001456854968"></a>

`AscendIndexInt8` requires the corresponding `AscendIndexInt8Config` to initialize the associated resources.

`Member Description`<a name="section1372191465013"></a>

| Member | Type | Description |
|--|--|--|
| `deviceList` | `std::vector<int>` | Device-side device ID list. |
| `resourceSize` | `int64_t` | Preallocated memory pool size on the device side, in bytes. |

`API Description`<a name="section135441937164218"></a>

| API Definition | `AscendIndexInt8Config()` |
| --- | --- |
| Description | Default constructor of `AscendIndexInt8Config`. The default `deviceList` is `0`, which means Ascend AI Processor 0 on the NPU is used as the heterogeneous computing platform for AscendFaiss retrieval. The default resource pool size is used. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table012165162914"></a>

| API Definition | `AscendIndexInt8Config(std::initializer_list<int> devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8Config`. It creates an `AscendIndexInt8Config` and configures device-side Ascend AI Processor resources and the resource pool size according to the values in `devices`. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resources`: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `INDEX_INT8_DEFAULT_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs, and the maximum number is 64. The configured `resources` value must not exceed `16 * 1024 MB` (`16 * 1024 * 1024 * 1024` bytes). |

<a name="table9202719152913"></a>

| API Definition | `AscendIndexInt8Config(std::vector<int> devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8Config`. It creates an `AscendIndexInt8Config` and configures device-side Ascend AI Processor resources and the resource pool size according to the values in `devices`. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resources`: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `INDEX_INT8_DEFAULT_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs, and the maximum number is 64. The configured `resources` value must not exceed 16 \* 1024 MB (16 \* 1024 \* 1024 \* 1024 bytes). |

## `AscendIndexInt8Flat`<a name="ZH-CN_TOPIC_0000001506334741"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506615033"></a>

`AscendIndexInt8Flat` stores `INT8` feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexInt8Flat`<a name="ZH-CN_TOPIC_0000001456375168"></a>

| API Definition | `AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8Flat`. It creates an `AscendIndexInt8` with dimension `dims`. The dimension of the vector set managed by a single `Index` is unique. It configures device-side resources according to the values in `config`. |
| Input | `int dims`: Dimension of the feature vector set managed by `AscendIndexInt8`.<br>`faiss::MetricType metric`: Distance metric type used by `AscendIndex` when performing feature vector similarity retrieval.<br>`AscendIndexInt8FlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ {64, 128, 256, 384, 512, 768, 1024}. `metric` ∈ {`faiss::MetricType::METRIC_L2`, `faiss::MetricType::METRIC_INNER_PRODUCT`}. |

<a name="table08035919302"></a>

| API Definition | `AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8Flat`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexScalarQuantizer *index`: CPU-side `Index` resource.<br>`AscendIndexInt8FlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. It must be a pointer of the `faiss::IndexScalarQuantizer` type generated by the `copyTo` interface of `AscendIndexInt8Flat`. |

<a name="table11312020103012"></a>

| API Definition | `AscendIndexInt8Flat(const faiss::IndexIDMap *index, AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8Flat`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexIDMap *index`: CPU-side `Index` resource.<br>`AscendIndexInt8FlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. It must be a pointer of the `faiss::IndexIDMap` type generated by the `copyTo` interface of `AscendIndexInt8Flat`. |

<a name="table186285584308"></a>

| API Definition | `AscendIndexInt8Flat(const AscendIndexInt8Flat&) = delete;` |
| --- | --- |
| Description | Declare this `Index` copy constructor as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexInt8Flat&`: A constant `AscendIndexInt8Flat`. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table206471151315"></a>

| API Definition | `virtual ~AscendIndexInt8Flat();` |
| --- | --- |
| Description | Destructor of `AscendIndexInt8Flat`. It destroys the `AscendIndexInt8Flat` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456375340"></a>

| API Definition | `void copyFrom(const faiss::IndexIDMap* index);` |
| --- | --- |
| Description | Copy an existing `index` to Ascend based on `AscendIndexInt8Flat`, and keep the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexIDMap *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. The dimension `d` parameter of the member index of this `Index` must be in the range {64, 128, 256, 384, 512, 768, 1024}. The total number of base vectors must satisfy `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. |

<a name="table862731073217"></a>

| API Definition | `void copyFrom(const faiss::IndexScalarQuantizer* index);` |
| --- | --- |
| Description | Copy an existing `index` to Ascend based on `AscendIndexInt8Flat`, and keep the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexScalarQuantizer* index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The dimension `d` parameter of the `Index` must be in the range {64, 128, 256, 384, 512, 768, 1024}. The total number of base vectors must satisfy `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001506334805"></a>

| API Definition | `void copyTo(faiss::IndexScalarQuantizer* index) const;` |
| --- | --- |
| Description | Copy the retrieval resources of `AscendIndexInt8Flat` to the CPU side. |
| Input | `faiss::IndexScalarQuantizer* index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The resources occupied by `Index` are freed by the user. |

<a name="table1981952413329"></a>

| API Definition | `void copyTo(faiss::IndexIDMap* index) const;`|
| --- | --- |
| Description | Copy the retrieval resources of `AscendIndexInt8Flat` to the CPU side. |
| Input | `faiss::IndexIDMap *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. The resources occupied by `Index` are freed by the user. |

### `getBase`<a name="ZH-CN_TOPIC_0000001506334753"></a>

| API Definition | `void getBase(int deviceId, std::vector<int8_t> &xb) const;` |
| --- | --- |
| Description | Get the feature vectors managed by this `AscendIndexInt8Flat` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | `std::vector<int8_t> &xb`: Base feature vectors stored by `AscendIndexInt8Flat` on `deviceId`. |
| Returns | None |
| Constraints | `deviceId` must be a valid device ID. |

### `getBaseSize`<a name="ZH-CN_TOPIC_0000001506414709"></a>

| API Definition | `size_t getBaseSize(int deviceId) const;` |
| --- | --- |
| Description | Get the number of feature vectors managed by this `AscendIndexInt8Flat` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | None |
| Returns | The number of feature vectors on the specified `deviceId`. |
| Constraints | `deviceId` must be a valid device ID. |

### `getIdxMap`<a name="ZH-CN_TOPIC_0000001506495853"></a>

| API Definition | `void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;` |
| --- | --- |
| Description | Get the feature vector IDs managed by this `AscendIndexInt8Flat` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | `std::vector<idx_t> &idxMap`: Base feature vector IDs stored by `AscendIndexInt8Flat` on `deviceId`. |
| Returns | None |
| Constraints | `deviceId` must be a valid device ID. |

### `operator =`<a name="ZH-CN_TOPIC_0000001506414909"></a>

| API Definition | `AscendIndexInt8Flat& operator=(const AscendIndexInt8Flat&) = delete;` |
| --- | --- |
| Description | Declare this `Index` assignment operator as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexInt8Flat&`: A constant `AscendIndexInt8Flat`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `reset`<a name="ZH-CN_TOPIC_0000001506495889"></a>

| API Definition | `void reset();` |
| --- | --- |
| Description | Clear the base vectors in this `AscendIndexInt8Flat`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `search_with_masks`<a name="ZH-CN_TOPIC_0000001456694912"></a>

| API Definition | `void search_with_masks(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;` |
| --- | --- |
| Description | Implement the feature vector search interface for `AscendIndexInt8`, and return the distances and IDs of the `k` most similar features based on the input feature vectors and the `mask`. The mask is a `0`/`1` bit string. Each bit indicates whether the corresponding feature in the base vector set participates in distance computation. `1` means participate, and `0` means not participate. |
| Input | `idx_t n`: Number of query feature vectors.<br>`const int8_t* x`: Feature vector data.<br>`idx_t k`: Number of most similar results to return.<br>`const void* mask`: Base vector set filter mask. |
| Output | `float *distances`: Distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: IDs of the top `k` nearest vectors to the query. |
| Returns | None |
| Constraints | The value range of `n` is `0 < n < 1e9`. `k` is usually not allowed to exceed `4096`. `x` must be a non-null pointer, and its length should be `dims * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `distances` and `labels` must be non-null pointers, and their lengths should be `k * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `mask` must be a non-null pointer, and the length of the passed mask must be `ntotal / 8 * n` (`ntotal` is the number of vectors in the base vector set). The mask is set in the order of the base vector set. If `remove_ids` is called before this interface, the order of base vectors changes. Therefore, call `getIdxMap` first to obtain the IDs of the base vectors, and then set the mask. This interface requires the base vector set to be stored on a single device. Otherwise, the filtering result may be incorrect. |

### `setPageSize`<a name="ZH-CN_TOPIC_0000002007453769"></a>

| API Definition | `void setPageSize(uint16_t pageBlockNum);` |
| --- | --- |
| Description | Set the number of base-vector blocks that this `AscendIndexInt8Flat` computes consecutively in one `search` call. |
| Input | `uint16_t pageBlockNum`: Number of base-vector blocks to compute consecutively in one call. If you do not set this parameter, the default is to compute 16 blocks consecutively at a time. The size of one block is determined by `blockSize` in `AscendIndexInt8FlatConfig`. The larger the value, the more memory `search` uses. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `pageBlockNum` is `0 < pageBlockNum ≤ 144`. This interface is mainly used for large base vector set scenarios and for performance tuning of the `search` interface. The larger the value, the more preallocated memory configured by `resourceSize` in `AscendIndexInt8FlatConfig` it consumes. You are advised to request enough preallocated memory first and then use this interface to tune parameters. |

## `AscendIndexInt8FlatConfig`<a name="ZH-CN_TOPIC_0000001456535040"></a>

`AscendIndexInt8Flat` requires the corresponding `AscendIndexInt8FlatConfig` to initialize the associated resources.

`Member Description`<a name="section1372191465013"></a>

| Member | Type | Description |
|--|--|--|
| `dIndexMode` | `Int8IndexMode` | Configures the INT8 retrieval mode for the `Index`. |
| `dBlockSize` | `uint32_t` | Configures the device-side `blockSize`. |

`API Description`<a name="section136272015172914"></a>

| API Definition | `AscendIndexInt8FlatConfig(uint32_t blockSize =BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8FlatConfig`. It creates an `AscendIndexInt8FlatConfig`, configures the device-side `blockSize`, and configures the INT8 retrieval mode. |
| Input | `uint32_t blockSize`: Configures the device-side `blockSize`. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value `BLOCK_SIZE` is `16384 * 16 = 262144`.<br>`Int8IndexMode indexMode`: Configures the INT8 retrieval mode for the `Index`. The default value is `DEFAULT_MODE`.<br> `DEFAULT_MODE`: Default mode. `PIPE_SEARCH_MODE`: This mode is optimized for scenarios where the batch is greater than or equal to `128`. When you use this mode, you are advised to set `resourceSize` to at least `1324 MB`. `WITHOUT_NORM_MODE`: This mode is not supported at this time. |
| Output | None |
| Returns | None |
| Constraints | The set of valid `blockSize` values is `{16384, 32768, 65536, 131072, 262144}`. In `PIPE_SEARCH_MODE`, `AscendIndexInt8Flat` supports only `METRIC_L2`. |

<a name="table1258103643012"></a>

| API Definition | `AscendIndexInt8FlatConfig(std::initializer_list<int> devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE);` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8FlatConfig`. It creates an `AscendIndexInt8FlatConfig` and configures device-side Ascend AI Processor resources and the resource pool size according to the values in `devices`. It also configures the device-side `blockSize` and the INT8 retrieval mode. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `INT8_FLAT_DEFAULT_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br>`uint32_t blockSize`: Configures the device-side `blockSize`. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value `BLOCK_SIZE` is `16384 * 16 = 262144`.<br>`Int8IndexMode indexMode`: Configures the INT8 retrieval mode for the `Index`. The default value is `DEFAULT_MODE`.<br> `DEFAULT_MODE`: Default mode. `PIPE_SEARCH_MODE`: This mode is optimized for scenarios where the batch is greater than or equal to `128`. When you use this mode, you are advised to set `resourceSize` to at least `1324 MB`. `WITHOUT_NORM_MODE`: This mode is not supported at this time. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs, and the maximum number is 64. The configured `resourceSize` value must not exceed `16 * 1024 MB` (`16 * 1024 * 1024 * 1024` bytes). When the batch is greater than or equal to `96`, you are advised to set `resourceSize` to at least `2 * 1024 MB` to improve algorithm performance. The set of valid `blockSize` values is `{16384, 32768, 65536, 131072, 262144}`. In `PIPE_SEARCH_MODE`, `AscendIndexInt8Flat` supports only `METRIC_L2`. |

<a name="table8629135217302"></a>

| API Definition | `AscendIndexInt8FlatConfig(std::vector<int> devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM, uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)` |
| --- | --- |
| Description | Constructor of `AscendIndexInt8FlatConfig`. It creates an `AscendIndexInt8FlatConfig` and configures device-side Ascend AI Processor resources and the resource pool size according to the values in `devices`. It also configures the device-side `blockSize` and the INT8 retrieval mode. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation, and it helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is `INT8_FLAT_DEFAULT_MEM` in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br>`uint32_t blockSize`: Configures the device-side `blockSize`. It constrains the amount of data processed by the tik operator in one calculation and the size of vectors stored in each shard of the base vector set. The default value `BLOCK_SIZE` is `16384 * 16 = 262144`.<br>`Int8IndexMode indexMode`: Configures the INT8 retrieval mode for the `Index`. The default value is `DEFAULT_MODE`.<br> `DEFAULT_MODE`: Default mode. `PIPE_SEARCH_MODE`: This mode is optimized for scenarios where the batch is greater than or equal to `128`. When you use this mode, you are advised to set `resourceSize` to at least `1324 MB`. `WITHOUT_NORM_MODE`: This mode is not supported at this time. |
| Output | None |
| Returns | None |
| Constraints | `devices` must be valid, unique device IDs, and the maximum number is 64. The configured `resourceSize` value must not exceed `16 * 1024 MB` (`16 * 1024 * 1024 * 1024` bytes). When the batch is greater than or equal to `96`, you are advised to set `resourceSize` to at least `2 * 1024 MB` to improve algorithm performance. The set of valid `blockSize` values is `{16384, 32768, 65536, 131072, 262144}`. In `PIPE_SEARCH_MODE`, `AscendIndexInt8Flat` supports only `METRIC_L2`. |

## `AscendIndexFlat`<a id="ZH-CN_TOPIC_0000001506334757"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506334829"></a>

`AscendIndexFlat` is the most basic feature retrieval algorithm. It stores FP16 floating-point feature vectors and performs brute-force search.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval uses OMP internally for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> `AscendIndexFlat` supports online operator conversion for L2 and IP distances. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (set it with `export MX_INDEX_USE_ONLINEOP=1`), the operator is converted and called online. To use online operators, the application must explicitly call `(void)aclFinalize()` at the end. You also need to include the header file `#include "acl/acl.h"`.

### `AscendIndexFlat`<a name="ZH-CN_TOPIC_0000001456375308"></a>

| API Definition | `AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexFlat`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexFlat *index`: CPU-side `Index` resource.<br>`AscendIndexFlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The dimension `d` parameter of this `Index` must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. |

<a name="table1735274911381"></a>

| API Definition | `AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexFlat`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexIDMap *index`: CPU-side `Index` resource.<br>`AscendIndexFlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. The dimension `d` parameter of this `Index` must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. |

<a name="table142416323911"></a>

| API Definition | `AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());` |
| --- | --- |
| Description | Constructor of `AscendIndexFlat`. It creates an `AscendIndexFlat` with dimension `dims`. The dimension of the vector set managed by a single `Index` is unique. It configures device-side resources according to the values in `config`. |
| Input | `int dims`: Dimension of the feature vector set managed by `AscendIndex`.<br>`faiss::MetricType metric`: Distance metric type used by `AscendIndexFlat` when performing feature vector similarity retrieval.<br>`AscendIndexFlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. `metric` ∈ {`faiss::MetricType::METRIC_L2`, `faiss::MetricType::METRIC_INNER_PRODUCT`}. |

<a name="table5169814143913"></a>

| API Definition | `AscendIndexFlat(const AscendIndexFlat&) = delete;` |
| --- | --- |
| Description | Declare this `Index` copy constructor as deleted, which means that the type is non-copyable. |
| Input | `const AscendIndexFlat&`: A constant `AscendIndexFlat`. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table04891725153918"></a>

| API Definition | `virtual ~AscendIndexFlat();` |
| --- | --- |
| Description | Destructor of `AscendIndexFlat`. It destroys the `AscendIndexFlat` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456535180"></a>

| API Definition | `void copyFrom(const faiss::IndexFlat *index);` |
| --- | --- |
| Description | Copy an existing `Index` to Ascend based on `AscendIndexFlat`, clear the current base vector set in `AscendIndexFlat`, and keep the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexFlat *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The dimension `d` parameter of this `Index` must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. |

<a name="table525914213409"></a>

| API Definition | `void copyFrom(const faiss::IndexIDMap *index);` |
| --- | --- |
| Description | Copy an existing `index` to Ascend based on `AscendIndexFlat`, clear the current base vector set in `AscendIndexFlat`, and keep the original device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexIDMap *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. Otherwise, the program may crash or the function may become unavailable. The dimension `d` parameter of this `Index` must be in the range {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. The total number of base vectors must satisfy `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001456535148"></a>

| API Definition | `void copyTo(faiss::IndexFlat *index) const;` |
| --- | --- |
| Description | Copies the retrieval resources of `AscendIndexFlat` to the CPU side. |
| Input | `faiss::IndexFlat *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The user must free the memory occupied by the `Index`. |

<a name="table154531752144016"></a>

| API Definition | `void copyTo(faiss::IndexIDMap *index) const;` |
| --- | --- |
| Description | Copies the retrieval resources of `AscendIndexFlat` to the CPU side. |
| Input | `faiss::IndexIDMap *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. The user must free the memory occupied by the `Index`. |

### `getBase`<a name="ZH-CN_TOPIC_0000001456375236"></a>

| API Definition | `void getBase(int deviceId, char* xb) const;` |
| --- | --- |
| Description | Gets the feature vectors managed by this `AscendIndexFlat` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | `char* xb`: The base library feature vectors stored by `AscendIndexFlat` on `deviceId`. |
| Returns | None |
| Constraints | `deviceId` must be a valid device ID.<br>`xb` must be a non-null pointer, and its length must be `dims * BaseSize * sizeof(float32)` bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `BaseSize` is the return value of `getBaseSize`. |

### `getBaseSize`<a name="ZH-CN_TOPIC_0000001456854956"></a>

| API Definition | `size_t getBaseSize(int deviceId) const;` |
| --- | --- |
| Description | Gets the number of feature vectors managed by this `AscendIndexFlat` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | None |
| Returns | The number of feature vectors on the specified `deviceId`. |
| Constraints | `deviceId` must be a valid device ID. |

### `getIdxMap`<a name="ZH-CN_TOPIC_0000001506334785"></a>

| API Definition | `void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;` |
| --- | --- |
| Description | Gets the feature vector IDs managed by this `AscendIndexFlat` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | `std::vector<idx_t> &idxMap`: The base library feature vector IDs stored by `AscendIndexFlat` on `deviceId`. |
| Returns | None |
| Constraints | `deviceId` must be a valid device ID. |

### `operator=`<a name="ZH-CN_TOPIC_0000001506495701"></a>

| API Definition | `AscendIndexFlat& operator=(const AscendIndexFlat&) = delete;` |
| --- | --- |
| Description | Declares the assignment operator as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexFlat&`: A constant `AscendIndexFlat`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `search_with_masks`<a name="ZH-CN_TOPIC_0000001810529650"></a>

| API Definition | `void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;` |
| --- | --- |
| Description | The feature vector query API of `AscendIndexFlat`. It returns the IDs of the `k` most similar features based on the input feature vectors. `mask` is a bit string of `0`s and `1`s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. `1` means participate, and `0` means do not participate. |
| Input | `idx_t n`: The number of query feature vectors.<br>`const float *x`: Feature vector data.<br>`idx_t k`: The number of most similar results to return.<br>`const void *mask`: Feature library mask. |
| Output | `float *distances`: The distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: The IDs of the top `k` nearest vectors for the query. |
| Returns | None |
| Constraints | The value of `n` must satisfy `0 < n < 1e9`. `k` is usually not allowed to exceed `4096`. `x` must be a non-null pointer, and its length must be `dim * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `distances` and `labels` must be non-null pointers, and each must have a length of `k * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `mask` must be a non-null pointer, and its length must be `n * ceil(ntotal / 8)`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `ntotal` is the number of base library features. `mask` is set according to the order of the base library. If you call `remove_ids` to delete feature vectors before calling this API, the order of the base library features changes. First call `getIdxMap` to obtain the IDs of the base library features, and then set `mask`. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect. |

<a name="table0628133121511"></a>

| API Definition | `void search_with_masks(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;` |
| --- | --- |
| Description | The feature vector query API of `AscendIndexFlat`. It returns the IDs of the `k` most similar features based on the input feature vectors. `mask` is a bit string of `0`s and `1`s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. `1` means participate, and `0` means do not participate. |
| Input | `idx_t n`: The number of query feature vectors.<br>`const uint16_t *x`: Feature vector data.<br>`idx_t k`: The number of most similar results to return.<br>`const void *mask`: Feature library mask. |
| Output | `float *distances`: The distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: The IDs of the top `k` nearest vectors for the query. |
| Returns | None |
| Constraints | The value of `n` must satisfy `0 < n < 1e9`. `k` is usually not allowed to exceed `4096`. `x` must be a non-null pointer, and its length must be `dim * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `distances` and `labels` must be non-null pointers, and each must have a length of `k * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `mask` must be a non-null pointer, and its length must be `n * ceil(ntotal / 8)`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `ntotal` is the number of base library features. `mask` is set according to the order of the base library. If you call `remove_ids` to delete feature vectors before calling this API, the order of the base library features changes. First call `getIdxMap` to obtain the IDs of the base library features, and then set `mask`. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect. |

## `AscendIndexFlatConfig`<a name="ZH-CN_TOPIC_0000001456375216"></a>

`AscendIndexFlat` requires the corresponding `AscendIndexFlatConfig` to initialize the corresponding resources.

**API Description**<a name="section140920164419"></a>

| API Definition | `inline AscendIndexFlatConfig()` |
| --- | --- |
| Description | The default constructor of `AscendIndexFlatConfig`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table46951722104415"></a>

| API Definition | `inline AscendIndexFlatConfig(std::initializer_list<int> devices, int64_t resourceSize = FLAT_DEFAULT_MEM)` |
| --- | --- |
| Description | The constructor of `AscendIndexFlatConfig`. It creates an `AscendIndexFlatConfig` and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in `devices`. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is `FLAT_DEFAULT_MEM` in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to `4194304` and the batch size is greater than or equal to `16`, use the following recommendations.<br>When the distance type of `AscendIndexFlat` is `faiss::METRIC_L2`, the recommended value is `1024 MB`. When the distance type of `AscendIndexFlat` is `faiss::METRIC_INNER_PRODUCT`, the recommended value is `1280 MB`. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, unique device IDs. The maximum number is `64`. The value configured for `resourceSize` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). When this value is set to `-1`, the Device-side Ascend AI Processor resource is configured to the default value `128 MB`. |

<a name="table842319354444"></a>

| API Definition | `inline AscendIndexFlatConfig(std::vector<int> devices, int64_t resourceSize = FLAT_DEFAULT_MEM)` |
| --- | --- |
| Description | The constructor of `AscendIndexFlatConfig`. It creates an `AscendIndexFlatConfig` and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in `devices`. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is `FLAT_DEFAULT_MEM` in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to `4194304` and the batch size is greater than or equal to `16`, use the following recommendations.<br>When the distance type of `AscendIndexFlat` is `faiss::METRIC_L2`, the recommended value is `1024 MB`. When the distance type of `AscendIndexFlat` is `faiss::METRIC_INNER_PRODUCT`, the recommended value is `1280 MB`. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, unique device IDs. The maximum number is `64`. The value configured for `resourceSize` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). When this value is set to `-1`, the Device-side Ascend AI Processor resource is configured to the default value `128 MB`. |

## `AscendIndexFlatL2`<a name="ZH-CN_TOPIC_0000001456375424"></a>

### Overview<a name="ZH-CN_TOPIC_0000001877955534"></a>

`AscendIndexFlatL2` is a brute-force feature retrieval algorithm that stores FP16 floating-point values and uses the L2 distance.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

> [!NOTE]
> The `AscendIndexFlatL2` algorithm supports online operator conversion. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (`export MX_INDEX_USE_ONLINEOP=1`), it converts the operators online and calls them. To use online operators, the user must explicitly call `(void)aclFinalize()` at the end of the application. The header file `#include "acl/acl.h"` is required.

### `AscendIndexFlatL2`<a name="ZH-CN_TOPIC_0000001506495761"></a>

<a name="zh-cn_topic_0000001294312541_table7235918388"></a>

| API Definition | `AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());` |
| --- | --- |
| Description | The constructor of `AscendIndexFlatL2`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `faiss::IndexFlatL2 *index`: CPU-side `Index` resource.<br>`AscendIndexFlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The value range of the `d` dimension parameter of the `Index` is `{32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}`. The value range of the total number of base library vectors is `0 ≤ n < 1e9`. The `metric_type` parameter must be `faiss::MetricType::METRIC_L2`. |

<a name="zh-cn_topic_0000001294591937_table7235918388"></a>

| API Definition | `AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());` |
| --- | --- |
| Description | The constructor of `AscendIndexFlatL2`. It creates an `AscendIndexFlatL2` with dimension `dims`. The dimension of a vector set managed by one `Index` is unique. It then sets Device-side resources according to the values configured in `config`. |
| Input | `int dims`: The dimension of a set of feature vectors managed by `AscendIndexFlatL2`.<br>`AscendIndexFlatConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584} |

<a name="zh-cn_topic_0000001247793230_table7235918388"></a>

| API Definition | `AscendIndexFlatL2(const AscendIndexFlatL2&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexFlatL2&`: A constant `AscendIndexFlatL2`. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="zh-cn_topic_0000001294312453_table7235918388"></a>

| API Definition | `virtual ~AscendIndexFlatL2()` |
| --- | --- |
| Description | The destructor of `AscendIndexFlatL2`. It destroys the `AscendIndexFlatL2` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001456375400"></a>

<a name="zh-cn_topic_0000001248112146_table7235918388"></a>

| API Definition | `void copyFrom(faiss::IndexFlat *index);` |
| --- | --- |
| Description | Copies an existing `index` to Ascend based on `AscendIndexFlat`, clears the current base library of `AscendIndexFlatL2`, and keeps the existing Device-side resource configuration of `AscendIndex`. |
| Input | `const faiss::IndexFlat *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The value range of the `d` dimension parameter of the `Index` is `{64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}`. The value range of the total number of base library vectors is `0 <= n < 1e9`. The `metric_type` parameter must be `faiss::MetricType::METRIC_L2`. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001456535052"></a>

<a name="zh-cn_topic_0000001247793178_table7235918388"></a>

| API Definition | `void copyTo(faiss::IndexFlat *index);` |
| --- | --- |
| Description | Copies the retrieval resources of `AscendIndexFlatL2` to the CPU side. |
| Input | `faiss::IndexFlat *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The user must free the memory occupied by the `Index`. |

### `operator=`<a name="ZH-CN_TOPIC_0000001456695116"></a>

<a name="zh-cn_topic_0000001294432513_table7235918388"></a>

| API Definition | `AscendIndexFlatL2& operator=(const AscendIndexFlatL2&) = delete;` |
| --- | --- |
| Description | Declares the assignment operator as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexFlatL2&`: A constant `AscendIndexFlatL2`. |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendIndexSQ`<a name="ZH-CN_TOPIC_0000001506614969"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456695120"></a>

`AscendIndexSQ` performs Scalar Quantization on the input vectors.

The vectors stored in the base library and the query vectors of each API must be normalized float values.

It supports multithreaded concurrent calls. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

### `AscendIndexSQ`<a name="ZH-CN_TOPIC_0000001506614933"></a>

| API Definition | `AscendIndexSQ(const faiss::IndexScalarQuantizer* index, AscendIndexSQConfig config = AscendIndexSQConfig());` |
| --- | --- |
| Description | The constructor of `AscendIndexSQ`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexScalarQuantizer* index`: CPU-side `Index` resource.<br>`AscendIndexSQConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The value range of the `d` dimension parameter of the `Index` is `{64, 128, 256, 384, 512, 768}`. The value range of the total number of base library vectors is `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. The `sq.qtype` parameter supports only `ScalarQuantizer::QuantizerType::QT_8bit`. |

<a name="table207325212487"></a>

| API Definition | `AscendIndexSQ(const faiss::IndexIDMap* index, AscendIndexSQConfig config = AscendIndexSQConfig());` |
| --- | --- |
| Description | The constructor of `AscendIndexSQ`. It creates a retrieval `Index` on Ascend based on an existing `index`. |
| Input | `const faiss::IndexIDMap* index`: CPU-side `Index` resource.<br>`AscendIndexSQConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The value range of the dimension parameter `d` of the member index is `{64, 128, 256, 384, 512, 768}`. The value range of the total number of base library vectors is `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. The `sq.qtype` parameter supports only `ScalarQuantizer::QuantizerType::QT_8bit`. |

<a name="table1132217014918"></a>

| API Definition | `AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexSQConfig config = AscendIndexSQConfig());` |
| --- | --- |
| Description | The constructor of `AscendIndexSQ`. It creates an `AscendIndex` with dimension `dims`. The dimension of a vector set managed by one `Index` is unique. It then sets Device-side resources according to the values configured in `config`. |
| Input | `int dims`: The dimension of a set of feature vectors managed by `AscendIndexSQ`.<br>`faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit`: Currently, only `ScalarQuantizer::QuantizerType::QT_8bit` is supported.<br>`faiss::MetricType metric`: The distance metric type used by `AscendIndex` when it performs feature vector similarity retrieval.<br>`AscendIndexSQConfig config`: Device-side resource configuration. |
| Output | None |
| Returns | None |
| Constraints | `dims` ∈ {64, 128, 256, 384, 512, 768}. `metric` ∈ {`faiss::MetricType::METRIC_L2`, `faiss::MetricType::METRIC_INNER_PRODUCT`}. |

<a name="table16655810104919"></a>

| API Definition | `AscendIndexSQ(const AscendIndexSQ&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexSQ&`: An `AscendIndexSQ` object. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table17704194534915"></a>

| API Definition | `virtual ~AscendIndexSQ();` |
| --- | --- |
| Description | The destructor of `AscendIndexSQ`. It destroys the `AscendIndexSQ` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `copyFrom`<a name="ZH-CN_TOPIC_0000001506615037"></a>

| API Definition | `void copyFrom(const faiss::IndexScalarQuantizer* index);` |
| --- | --- |
| Description | Copies an existing `index` to Ascend based on `AscendIndexSQ`, clears the current base library of `AscendIndexSQ`, and keeps the existing Device-side resource configuration of `AscendIndexSQ`. |
| Input | `const faiss::IndexScalarQuantizer* index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The value range of the `d` dimension parameter of the `Index` is `{64, 128, 256, 384, 512, 768}`. The value range of the total number of base library vectors is `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. The `sq.qtype` parameter supports only `ScalarQuantizer::QuantizerType::QT_8bit`. |

<a name="table853716365015"></a>

| API Definition | `void copyFrom(const faiss::IndexIDMap* index);` |
| --- | --- |
| Description | Copies an existing `index` to Ascend based on `AscendIndexSQ`, clears the current base library of `AscendIndexSQ`, and keeps the existing Device-side resource configuration of `AscendIndexSQ`. |
| Input | `const faiss::IndexIDMap *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. The value range of the dimension parameter `d` of the member index is `{64, 128, 256, 384, 512, 768}`. The value range of the total number of base library vectors is `0 ≤ n < 1e9`. The `metric_type` parameter must be one of `{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}`. The `sq.qtype` parameter supports only `ScalarQuantizer::QuantizerType::QT_8bit`. |

### `copyTo`<a name="ZH-CN_TOPIC_0000001456695084"></a>

| API Definition | `void copyTo(faiss::IndexScalarQuantizer* index) const;` |
| --- | --- |
| Description | Copies the retrieval resources of `AscendIndexSQ` to the CPU side. |
| Input | `faiss::IndexScalarQuantizer* index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid CPU `Index` pointer. The user must free the memory occupied by the `Index`. |

<a name="table817201512500"></a>

| API Definition | `void copyTo(faiss::IndexIDMap* index) const;` |
| --- | --- |
| Description | Copies the retrieval resources of `AscendIndexSQ` to the CPU side. |
| Input | `faiss::IndexIDMap *index`: CPU-side `Index` resource. |
| Output | None |
| Returns | None |
| Constraints | `index` must be a valid `IndexIDMap` pointer. The user must free the memory occupied by the `Index`. |

### `getBase`<a name="ZH-CN_TOPIC_0000001456694928"></a>

| API Definition | `void getBase(int deviceId, char* xb) const;` |
| --- | --- |
| Description | Gets the feature vectors managed by this `AscendIndexSQ` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | `char* xb`: The base library feature vectors stored by `AscendIndexSQ` on `deviceId`. |
| Returns | None |
| Constraints | `deviceId` must be a valid device ID. `xb` must be a non-null pointer, and its length must be `dims * BaseSize * sizeof(uint8_t)` bytes. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `BaseSize` is the return value of `getBaseSize`. |

### `getBaseSize`<a name="ZH-CN_TOPIC_0000001456854788"></a>

| API Definition | `size_t getBaseSize(int deviceId) const;` |
| --- | --- |
| Description | Gets the number of feature vectors managed by this `AscendIndexSQ` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | None |
| Returns | The number of feature vectors on the specified `deviceId`. |
| Constraints | `deviceId` must be a valid device ID. |

### `getIdxMap`<a name="ZH-CN_TOPIC_0000001456375152"></a>

| API Definition | `void getIdxMap(int deviceId, std::vector<idx_t>& idxMap) const;` |
| --- | --- |
| Description | Gets the feature vector IDs managed by this `AscendIndexSQ` on the specified `deviceId`. |
| Input | `int deviceId`: Device-side device ID. |
| Output | `std::vector<idx_t> &idxMap`: The base library feature vector IDs stored by `AscendIndexSQ` on `deviceId`. |
| Returns | None |
| Constraints | `deviceId` must be a valid device ID. |

### `operator=`<a name="ZH-CN_TOPIC_0000001456375300"></a>

| API Definition | `AscendIndexSQ& operator=(const AscendIndexSQ&) = delete;` |
| --- | --- |
| Description | Declares the assignment operator as deleted. In other words, this is a non-copyable type. |
| Input | `const AscendIndexSQ&`: An `AscendIndexSQ` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `search_with_filter`<a name="ZH-CN_TOPIC_0000001810589742"></a>

| API Definition | `void search_with_filter(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters) const;` |
| --- | --- |
| Description | The feature vector query API of `AscendIndexSQ`. It returns the IDs of the `k` most similar features based on the input feature vectors. It also provides CID-based filtering. `filters` is a `uint32_t` array of length `n * 6`. Every six `uint32_t` values form one filter. The first four values of each filter, that is, 128 bits, represent the corresponding CID. The last two values represent the left-closed timestamp interval, that is, [`x`, `y`). |
| Input | `idx_t n`: The number of query feature vectors.<br>`const float *x`: Feature vector data.<br>`idx_t k`: The number of most similar results to return.<br>`const void *filters`: Filter conditions. |
| Output | `float *distances`: The distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: The IDs of the top `k` nearest vectors for the query. |
| Returns | None |
| Constraints | The value of `n` must satisfy `0 < n < 1e9`. `k` is usually not allowed to exceed `4096`. `x` must be a non-null pointer, and its length must be `dims * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `distances` and `labels` must be non-null pointers, and each must have a length of `k * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `filters` must be a non-null pointer to a `uint32_t` array of length `n * 6`. Otherwise, out-of-bounds read errors may occur and cause the program to crash. |

### `search_with_masks`<a name="ZH-CN_TOPIC_0000001456694932"></a>

| API Definition | `void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;` |
| --- | --- |
| Description | The feature vector query API of `AscendIndexSQ`. It returns the IDs of the `k` most similar features based on the input feature vectors. `mask` is a bit string of `0`s and `1`s. Each bit indicates whether the feature at the corresponding position in the base library participates in distance calculation. `1` means participate, and `0` means do not participate. |
| Input | `idx_t n`: The number of query feature vectors.<br>`const float *x`: Feature vector data.<br>`idx_t k`: The number of most similar results to return.<br>`const void *mask`: Feature library mask. |
| Output | `float *distances`: The distance values between the query vectors and the top `k` nearest vectors.<br>`idx_t *labels`: The IDs of the top `k` nearest vectors for the query. |
| Returns | None |
| Constraints | The value of `n` must satisfy `0 < n < 1e9`. `k` is usually not allowed to exceed `4096`. `x` must be a non-null pointer, and its length must be `dims * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `distances` and `labels` must be non-null pointers, and each must have a length of `k * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `mask` must be a non-null pointer, and its length must be `n * ceil(ntotal / 8)`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. `ntotal` is the number of base library features. `mask` is set according to the order of the base library. If you call `remove_ids` to delete feature vectors before calling this API, the order of the base library features changes. First call `getIdxMap` to obtain the IDs of the base library features, and then set `mask`. To use this API, the base library must be stored on one device. Otherwise, the filtering result may be incorrect. |

### `train`<a name="ZH-CN_TOPIC_0000001506414905"></a>

| API Definition | `void train(idx_t n, const float *x) override;` |
| --- | --- |
| Description | Trains the quantizer on `AscendIndexSQ`. This API inherits the interface from `AscendFaiss` and provides the concrete implementation. **Note that you must train the `Index` before you call `add`.** |
| Input | `idx_t n`: The number of feature vectors in the training set.<br>`const float *x`: Feature vector data. |
| Output | None |
| Returns | None |
| Constraints | The value of `n` must satisfy `0 < n < 1e9`. `x` must be a non-null pointer, and its length must be `dims * n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. Training collects the data distribution. A small training set may affect query accuracy. |

## `AscendIndexSQConfig`<a name="ZH-CN_TOPIC_0000001456375392"></a>

`AscendIndexSQ` requires the corresponding `AscendIndexSQConfig` to initialize its resources.

| API Definition | `inline AscendIndexSQConfig()` |
| --- | --- |
| Description | The default constructor of `AscendIndexSQConfig`. The default `deviceList` is `0`, which means the first Ascend AI Processor of the NPU is selected as the heterogeneous computing platform for `AscendFaiss` retrieval. The default resource pool size is used. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table108621239568"></a>

| API Definition | `inline AscendIndexSQConfig(std::initializer_list<int> devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)` |
| --- | --- |
| Description | The constructor of `AscendIndexSQConfig`. It creates an `AscendIndexSQConfig` and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in `devices`. |
| Input | `std::initializer_list<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is `SQ_DEFAULT_MEM` defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to `10,000,000` and the batch size is greater than or equal to `16`, you are advised to set it to `1024 MB`.<br>`uint32_t blockSize`: Configures the `blockSize` on the Device side. It constrains the amount of data processed in a single `tik` operator execution and the size of vectors stored in each shard of the base library. The default value is `16384 * 16 = 262144`. This value affects the maximum number of `Index` objects that can be created and retrieval performance. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, unique device IDs. The maximum number is `64`. The value configured for `resourceSize` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). When this value is set to `-1`, the Device-side Ascend AI Processor resource is configured to the default value `128 MB`. The valid values of `blockSize` are `{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}`. |

<a name="table1735412445711"></a>

| API Definition | `inline AscendIndexSQConfig(std::vector<int> devices, int64_t resourceSize = SQ_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)` |
| --- | --- |
| Description | The constructor of `AscendIndexSQConfig`. It creates an `AscendIndexSQConfig` and sets Device-side Ascend AI Processor resources and the resource pool size according to the values configured in `devices`. |
| Input | `std::vector<int> devices`: Device-side device IDs.<br>`int64_t resourceSize`: The preset memory pool size on the Device side, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is `SQ_DEFAULT_MEM` defined in the header file. This parameter is determined by the base library size and the search batch size. When the base library size is greater than or equal to `10,000,000` and the batch size is greater than or equal to `16`, you are advised to set it to `1024 MB`.<br>`uint32_t blockSize`: Configures the `blockSize` on the Device side. It constrains the amount of data processed in a single `tik` operator execution and the size of vectors stored in each shard of the base library. The default value is `16384 * 16 = 262144`. This value affects the maximum number of `Index` objects that can be created and retrieval performance. |
| Output | None |
| Returns | None |
| Constraints | `devices` must contain valid, unique device IDs. The maximum number is `64`. The value configured for `resourceSize` must not exceed `10 * 1024 MB` (`10 * 1024 * 1024 * 1024` bytes). When this value is set to `-1`, the Device-side Ascend AI Processor resource is configured to the default value `128 MB`. The valid values of `blockSize` are `{16384 * 8, 16384 * 16, 16384 * 32, 16384 * 64}`. |

## `IndexIL`<a name="ZH-CN_TOPIC_0000001506414825"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456535188"></a>

`IndexIL` is a feature management abstract class based on a contiguous memory allocation mechanism. It serves retrieval algorithms that use indices as labels. To use it, you must inherit from it and implement all interfaces.

The vectors stored in the base library and the query vectors of each API must be normalized FP16 floating-point values. (`IL` stands for "Indices as Labels".)

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, the user must lock before use. Otherwise, the retrieval APIs may raise exceptions. It also does not support sharing a Device across different threads.

### `AddFeatures`<a name="ZH-CN_TOPIC_0000001506414693"></a>

| API Definition | `virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;` |
| --- | --- |
| Description | Inserts `n` feature vectors with specified indices into the feature library. If a feature vector already exists at an index, this insertion is equivalent to an update. |
| Input | `int n`: The number of feature vectors to insert.<br>`const float16_t *features`: Feature vectors, with a length of `n * vector dimension dim`.<br>`const idx_t *indices`: The index values corresponding to the feature vectors, with a length of `n`. |
| Output | None |
| Returns | `APP_ERROR`: The return status of the call. For details, see the reference for API return values. |
| Constraints | The input parameters are constrained by the implementation class. `features` and `indices` must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### `IndexIL`<a name="ZH-CN_TOPIC_0000001456695020"></a>

| API Definition | `IndexIL();` |
| --- | --- |
| Description | The constructor of `IndexIL`. It creates a feature management object. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `~IndexIL`<a name="ZH-CN_TOPIC_0000001506334781"></a>

| API Definition | `virtual ~IndexIL();` |
| --- | --- |
| Description | The destructor of `IndexIL`. It destroys the feature management object. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `Finalize`<a name="ZH-CN_TOPIC_0000001456375356"></a>

| API Definition | `virtual APP_ERROR Finalize() = 0;` |
| --- | --- |
| Description | Releases the feature library management resources. |
| Input | None |
| Output | None |
| Returns | `APP_ERROR`: The return status of the call. For details, see the reference for API return values. |
| Constraints | None |

### `GetFeatures`<a name="ZH-CN_TOPIC_0000001506495833"></a>

| API Definition | `virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;` |
| --- | --- |
| Description | Queries the feature vectors for `n` specified index values. |
| Input | `int n`: The number of feature vectors to obtain.<br>`const idx_t *indices`: The index values to query, with a length of `n`. |
| Output | `float16_t *features`: The feature vectors corresponding to the queried indices, with a length of `n * vector dimension dim`. The user must allocate memory before the call and ensure that the memory size is correct. |
| Returns | `APP_ERROR`: The return status of the call. For details, see the reference for API return values. |
| Constraints | The input parameters are constrained by the implementation class. `features` and `indices` must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### `GetNTotal`<a name="ZH-CN_TOPIC_0000001456535092"></a>

| API Definition | `virtual int GetNTotal() const = 0;` |
| --- | --- |
| Description | Queries the maximum occupied space of the current feature library vectors.<br>Feature vectors are inserted starting from index `0`. If the inserted feature vector indices are continuous, `ntotal` equals the number of feature vectors. Otherwise, `ntotal` equals the maximum inserted index value plus `1`. For performance reasons, the operator batches memory operations and, by default, treats the space at and before the maximum index position as valid base library vectors and includes it in the calculation. The user must use this API to obtain the total number of base library entries recorded inside the `Index`, and then allocate the corresponding memory space to pass parameters to the corresponding functional APIs. For details, see the specific API. |
| Input | None |
| Output | None |
| Returns | `int ntotal`: See the description. |
| Constraints | None |

### `Init`<a name="ZH-CN_TOPIC_0000001506334657"></a>

| API Definition | `virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;` |
| --- | --- |
| Description | Initializes feature library parameters and allocates base library memory resources. |
| Input | `int dim`: Feature vector dimension.<br>`AscendMetricType metricType`: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br>`int capacity`: Maximum base library capacity. The allocated memory size is `capacity * dim * sizeof(float)` bytes.<br>`int resourceSize`: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling `aclrtmalloc` to allocate memory, which improves performance. The default value is `-1`, which means the cache resource is allocated with the default size of `128 MB`. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.<br>For example, if the query batch size is `64`, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set `resourceSize` to `64 * 1000000 * 4 = 256,000,000` bytes. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | The implementation class constrains the input parameters. |

### RemoveFeatures API<a name="ZH-CN_TOPIC_0000001456534932"></a>

| API Definition | `virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;` |
| --- | --- |
| Description | Deletes the feature vectors with the specified indices from the vector library. |
| Input | `int n`: Number of feature vectors to delete.<br>`const idx_t *indices`: Indices of the feature vectors. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | The implementation class constrains the input parameters. `indices` must be a non-null pointer, and its length must be `n`. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### SetNTotal API<a name="ZH-CN_TOPIC_0000001456375256"></a>

| API Definition | `virtual APP_ERROR SetNTotal(int n) = 0;` |
| --- | --- |
| Description | Provides an interface for adjusting the `ntotal` count externally.<br>After base library vectors are added, the `Index` internally updates the `ntotal` value according to the largest inserted index, but it does not record which regions in the range [0, `ntotal` ] are invalid. Therefore, the `RemoveFeatures` operation does not change the `ntotal` value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set `ntotal` manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, `ntotal = 100`. If you delete the base library entries with indices from 80 to 90, the `ntotal` value inside `Index` remains unchanged and can only be set to a value in [ `ntotal`, `capacity` ]. If you then delete the base library entries with indices from 90 to 99, you can manually set `ntotal` to a value in [80, `capacity` ]. When you set it to `80`, the amount of base library data involved in comparison decreases by 20 vectors. |
| Input | `int n`: Maximum base library index managed by the service side, plus 1. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | The implementation class constrains the input parameters. |

## IndexILFlat<a name="ZH-CN_TOPIC_0000001506614925"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506414785"></a>

`IndexILFlat` inherits from `IndexIL` and is a pure Device-side retrieval solution. It uses resources such as the Ascend AI Processor and AI Core to enable each API. The program must be compiled on the Host into a binary, and then the binary and related runtime dependencies are deployed to the Device for execution. `IndexILFlat` uses the [Init](#init) interface to initialize the specified resources. After initialization, it allocates a contiguous block of memory to store the base library. After use, call the [Finalize](#finalize) interface to release the resources.

`IndexILFlat` currently receives only functional and performance maintenance on Atlas Inference Series products. The base library and query vectors must be normalized by the user, and the interfaces currently support only the inner product distance. For details, see [IndexILFlat](#indexilflat). Successful execution of this algorithm depends on the OM file of the TIK operator. In a pure-Device scenario, ensure that the deployed OM file is generated from the Index SDK deliverable and has not been tampered with.

Multithreaded concurrent calls are supported. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

### AddFeatures API<a name="ZH-CN_TOPIC_0000001456854852"></a>

| API Definition | `APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;` |
| --- | --- |
| Description | Inserts `n` feature vectors with the specified indices into the feature library. If a feature vector already exists at an index, the API updates it. |
| Input | `int n`: Number of feature vectors to insert.<br>`const float16_t *features`: Feature vectors to insert. The length is `n * dim`, where `dim` is the vector dimension.<br>`const idx_t *indices`: Indices of the feature vectors to insert. The valid length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `capacity` ). `n`: The value must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### ComputeDistance API<a name="ZH-CN_TOPIC_0000001456535116"></a>

| API Definition | `APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Returns the distances between `n` feature vectors and all feature vectors in the base library. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped distances are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: External memory that stores the distances between query vectors and base library vectors. The total length should be `n * nTotalPad` (`ntotalPad` is `(*ntotal + 15) / 16 * 16`, that is, `ntotal` rounded up to a multiple of 16). |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `distances`: The required buffer length is `n * ntotalPad` (`ntotalPad` is `(*ntotal + 15) / 16 * 16`, that is, the result of rounding `ntotal` up to a multiple of 16. The valid comparison distances for each query are stored in the first `ntotal` positions, and the padded data has no practical meaning). If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `queries` and `distances` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### ComputeDistanceByIdx API<a name="ZH-CN_TOPIC_0000001456694920"></a>

| API Definition | `APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Similar to `ComputeDistance`, except that `ComputeDistance` calculates the distances between query vectors and all base library vectors, whereas `ComputeDistanceByIdx` calculates only the distances between query vectors and the base library vectors at the given indices. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped `topk` results are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The valid length is `n * dim`, and `dim` must match the dimension specified during initialization.<br>`const int *num`: Number of base library feature vectors to compare for each query. The length is `n`.<br>`const idx_t *indices`: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum `num` value. The length of `indices` is `n * max(num)`.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum `num` value. The total length is `n * max(num)`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in [0, `capacity` ]. `num`: User-specified length `n`, and each `num` value must be in [0, `ntotal`]. `indices`: Each feature index must be in [0, `ntotal` ). For example, if `n = 3` and `num[3] = {1, 3, 5}`, the three queries compare with 1, 3, and 5 base library vectors respectively. Since `max(num) = 5`, the storage space pointed to by `indices` is aligned to 5, and the total size is `3 * 5 * sizeof(idx_t)` bytes, for example, `{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}`. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### ComputeDistanceByThreshold API<a name="ZH-CN_TOPIC_0000001506615117"></a>

| API Definition | `APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Adds threshold filtering on top of `ComputeDistance` and returns only the distances that meet the threshold condition. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), `distances` contains the mapped results after threshold filtering. |
| Input | `int n`: Number of feature vectors to query.<br>`float16_t *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`float threshold`: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by `threshold`.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `int *num`: Number of base library vectors that meet the threshold condition for each query, with length `n`.<br>`idx_t *indices`: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to `topk`. The total length of `indices` is `n * topk`.<br>`float *distances`: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of `indices`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in [0, `capacity` ]. `topk`: The value must be in [0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### Finalize API<a name="ZH-CN_TOPIC_0000001506414845"></a>

| API Definition | `APP_ERROR Finalize() override;` |
| --- | --- |
| Description | Releases feature library management resources. |
| Input | None |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | None |

### GetFeatures API<a name="ZH-CN_TOPIC_0000001456854992"></a>

| API Definition | `APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;` |
| --- | --- |
| Description | Queries the feature vectors with the specified indices for `n` entries. |
| Input | `int n`: Number of base library vectors to get.<br>`const idx_t *indices`: Indices corresponding to the `n` base library vectors to get. |
| Output | `float16_t *features`: Feature vectors corresponding to the queried indices. The length is `n * dim`, where `dim` is the vector dimension. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### GetNTotal API<a name="ZH-CN_TOPIC_0000001456375336"></a>

| API Definition | `int GetNTotal() const override;` |
| --- | --- |
| Description | Queries the theoretical maximum number of feature vectors in the current feature library. If the feature vector indices are inserted consecutively, `ntotal` is equal to the number of feature vectors. |
| Input | None |
| Output | `int ntotal`: The theoretical maximum number of feature vectors, that is, the maximum base library index plus 1. |
| Returns | `int ntotal`: See the description. |
| Constraints | None |

### IndexILFlat API<a name="ZH-CN_TOPIC_0000001456694872"></a>

| API Definition | `IndexILFlat();` |
| --- | --- |
| Description | Constructor of `IndexILFlat`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table194381755582"></a>

| API Definition | `IndexILFlat(const IndexILFlat&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor of `IndexILFlat` as deleted. Therefore, `IndexILFlat` is a non-copyable type. |
| Input | `const IndexILFlat&`: `IndexILFlat` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~IndexILFlat` API<a name="ZH-CN_TOPIC_0000001456375172"></a>

<a name="table11904175418"></a>

| API Definition | `virtual ~IndexILFlat();` |
| --- | --- |
| Description | Destructor of `IndexILFlat`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### Init API<a name="ZH-CN_TOPIC_0000001456375212"></a>

| API Definition | `APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;` |
| --- | --- |
| Description | Initializes feature library parameters and allocates base library memory resources. |
| Input | `int dim`: Feature vector dimension.<br>`AscendMetricType metricType`: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br>`int capacity`: Maximum base library capacity. The API allocates `capacity * dim * sizeof(fp16)` bytes of memory based on the `capacity` value.<br>`int64_t resourceSize`: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling the `aclrtmalloc` interface to allocate memory, which improves performance.<br>The default value is `-1`, which means that the cache resource is allocated with the default size of `128 MB`. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.<br>For example, if the query batch size is `64`, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set `resourceSize` to `64 * 1000000 * 4 = 256,000,000` bytes. Note that the maximum cache resource supported by the interface is `4 GB`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `dim` ∈ {32, 64, 128, 256, 384, 512, 1024}. `metricType`: `IndexILFlat` currently implements only the inner product distance, so it supports only `AscendMetricType::ASCEND_METRIC_INNER_PRODUCT`. `capacity`: The maximum memory that the API can allocate for the base library is `12,288,000,000` bytes, and the allowed range of `capacity` is (0, 12000000]. For example, for a base library vector set with 512 dimensions and the FP16 type, the maximum supported `capacity` is 12 million (`12288000000 / (512 * sizeof(fp16))`). For a base library vector set with 256 dimensions and the FP16 type, `capacity` can still be set to at most 12 million, even though the memory limit supports a larger value. `resourceSize` can be set to `-1` or any value in [134217728, 4294967296], in bytes, which is equivalent to `[128 MB, 4096 MB]`. This parameter is determined jointly by the base library size and the search batch size. When the base library contains at least 10 million vectors and the batch size is at least 16, you are advised to set it to `1024 MB`. |

### `operator =` API<a name="ZH-CN_TOPIC_0000001897140809"></a>

| API Definition | `IndexILFlat& operator=(const IndexILFlat&) = delete;` |
| --- | --- |
| Description | Declares this `Index` assignment operator as deleted. Therefore, the type is non-copyable. |
| Input | `const IndexILFlat&`: `IndexILFlat` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### RemoveFeatures API<a name="ZH-CN_TOPIC_0000001506414837"></a>

| API Definition | `APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;` |
| --- | --- |
| Description | Deletes the feature vectors with the specified indices from the vector library. |
| Input | `int n`: Number of feature vectors to delete.<br>`const idx_t *indices`: Indices of the feature vectors. The length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `indices` must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### Search API<a name="ZH-CN_TOPIC_0000001456854856"></a>

| API Definition | `APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Returns the indices and corresponding distances of the `topk` base library vectors that are closest to the query vectors. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped distances are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`int topk`: Sorts the comparison distances between the query vectors and the base library and returns `topk` results.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: External memory. It stores the cosine distances corresponding to the `topk * n` base library feature vectors that are most similar to the query. The length is `n * topk`.<br>`idx_t *indices`: External memory. It returns the indices corresponding to the `topk` base library vectors that are most similar to the query. The length is `n * topk`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `topk`: The value must be in (0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, and `distances` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### SearchByThreshold API<a name="ZH-CN_TOPIC_0000001456694892"></a>

| API Definition | `APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Adds threshold filtering on top of `Search` and returns only the results that meet the threshold condition. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped `topk` results are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The length is `n * dim`.<br>`float threshold`: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by `threshold`.<br>`int topk`: Sorts the comparison distances between the query vectors and the base library and returns `topk` results.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `int *num`: Number of base library vectors that meet the threshold condition for each query. The length is `n`.<br>`idx_t* indices`: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to `topk`. The total length of `indices` is `n * topk`.<br>`float *distances`: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of `indices`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `topk`: The value must be in (0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### SetNTotal API<a name="ZH-CN_TOPIC_0000001456854892"></a>

| API Definition | `APP_ERROR SetNTotal(int n) override;` |
| --- | --- |
| Description | Provides an interface for adjusting the `ntotal` count externally.<br>After base library vectors are added, the `Index` internally updates the `ntotal` value according to the largest inserted index, but it does not record which regions in the range [0, `ntotal` ] are invalid. Therefore, the `RemoveFeatures` operation does not change the `ntotal` value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set `ntotal` manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, `ntotal = 100`. If you delete the base library entries with indices from 80 to 90, the `ntotal` value inside `Index` remains unchanged and can only be set to a value in [ `ntotal`, `capacity` ]. If you then delete the base library entries with indices from 90 to 99, you can manually set `ntotal` to a value in [80, `capacity` ]. When you set it to `80`, the amount of base library data involved in comparison decreases by 20 vectors. |
| Input | `int n`: Maximum base library index managed by the service side, plus 1. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in [0, `capacity` ]. |

## AscendIndexILFlat<a name="ZH-CN_TOPIC_0000002514896041"></a>

### Overview<a name="ZH-CN_TOPIC_0000002482656058"></a>

`AscendIndexILFlat` is the standard-mode scenario of `ILFlat`. You need to use `Init` to initialize the corresponding resources. After initialization, it allocates a contiguous block of memory to store the base library. After use, call the `Finalize` interface to release the resources.

`AscendIndexILFlat` supports only Atlas Inference Series products and only the inner product distance type in the standard deployment mode. `AscendIndexILFlat` depends on the Flat and AICPU operators. For details, see [Flat](../user_guide.md#generating-operators) and [AICPU](../user_guide.md#generating-operators).

Multithreaded concurrent calls are supported. You must set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to another value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

### AddFeatures API<a name="ZH-CN_TOPIC_0000002514776041"></a>

| API Definition | `APP_ERROR AddFeatures(int n, const float *features);` |
| --- | --- |
| Description | Adds `n` feature vectors to the feature library. |
| Input | `int n`: Number of feature vectors to insert.<br>`const float *features`: Feature vectors to insert. The length is `n * dim`, where `dim` is the vector dimension. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `features` must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table392463914228"></a>

| API Definition | `APP_ERROR AddFeatures(int n, const float16_t *features);` |
| --- | --- |
| Description | Adds `n` feature vectors to the feature library. |
| Input | `int n`: Number of feature vectors to insert.<br>`const float16_t *features`: Feature vectors to insert. The length is `n * dim`, where `dim` is the vector dimension. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `features` must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### AscendIndexILFlat API<a name="ZH-CN_TOPIC_0000002516511133"></a>

| API Definition | `AscendIndexILFlat();` |
| --- | --- |
| Description | Constructor of `AscendIndexILFlat`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table161511529133912"></a>

| API Definition | `AscendIndexILFlat(const AscendIndexILFlat&) = delete;` |
| --- | --- |
| Description | Declares the copy constructor of `AscendIndexILFlat` as deleted. Therefore, `AscendIndexILFlat` is a non-copyable type. |
| Input | `const AscendIndexILFlat&`: `AscendIndexILFlat` object. |
| Output | None |
| Returns | None |
| Constraints | None |

<a name="table62621513124018"></a>

| API Definition | `virtual ~AscendIndexILFlat();` |
| --- | --- |
| Description | Destructor of `AscendIndexILFlat`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### ComputeDistance API<a name="ZH-CN_TOPIC_0000002482736032"></a>

| API Definition | `APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Returns the distances between `n` feature vectors and all feature vectors in the base library. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped distances are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: External memory. It stores the distances between query vectors and base library vectors. The total length should be `n * nTotalPad` (`ntotalPad` is `(*ntotal + 15) / 16 * 16`, that is, `ntotal` rounded up to a multiple of 16). |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The recommended value should be in (0, `capacity` ]. `distances`: The required buffer length is `n * ntotalPad` (`ntotalPad` is `(*ntotal + 15) / 16 * 16`, that is, the result of rounding `ntotal` up to a multiple of 16. The valid comparison distances for each query are stored in the first `ntotal` positions, and the padded data has no practical meaning). If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `queries` and `distances` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table17574555124816"></a>

| API Definition | `APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Returns the distances between `n` feature vectors and all feature vectors in the base library. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped distances are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: External memory. It stores the distances between query vectors and base library vectors. The total length should be `n * nTotalPad` (`ntotalPad` is `(*ntotal + 15) / 16 * 16`, that is, `ntotal` rounded up to a multiple of 16). |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The recommended value should be in (0, `capacity` ]. `distances`: The required buffer length is `n * ntotalPad` (`ntotalPad` is `(*ntotal + 15) / 16 * 16`, that is, the result of rounding `ntotal` up to a multiple of 16. The valid comparison distances for each query are stored in the first `ntotal` positions, and the padded data has no practical meaning). If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `queries` and `distances` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### ComputeDistanceByIdx API<a name="ZH-CN_TOPIC_0000002514896043"></a>

| API Definition | `APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | `ComputeDistance` calculates the distances between query vectors and all base library vectors, whereas `ComputeDistanceByIdx` calculates only the distances between query vectors and the base library vectors at the specified indices. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped `topk` results are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float *queries`: Feature vectors to query. The valid length is `n * dim`, and `dim` must match the dimension specified during initialization.<br>`const int *num`: Number of base library feature vectors to compare for each query. The length is `n`.<br>`const idx_t *indices`: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum `num` value. The length of `indices` is `n * max(num)`. If the input is on the host, `indices` is a host pointer. If the input is on the device, `indices` is a device pointer.<br>`MEMORY_TYPE memoryType`: Policy for where the input and output are stored. The default is `MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST`. The available policies are as follows:<br>`MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST`: input on the host and output on the host. `MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE`: input on the device and output on the device. `MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST`: input on the device and output on the host. `MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE`: input on the host and output on the device.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum `num` value. The total length is `n * max(num)`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `num`: User-specified, with length `n`, and each query's `num` value must be in [0, `ntotal`]. `indices`: Each feature index must be in [0, `ntotal` ). For example, if `n = 3` and `num[3] = {1, 3, 5}`, the three queries compare with 1, 3, and 5 base library vectors respectively. Since `max(num) = 5`, the storage space pointed to by `indices` is aligned to 5, and the total size is `3 * 5 * sizeof(idx_t)` bytes, for example, `{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}`. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. When selecting a `memoryType` storage policy, `queries` and `distances` must be pointers to the corresponding location, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table93703718308"></a>

| API Definition | `APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices, float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | `ComputeDistance` calculates the distances between query vectors and all base library vectors, whereas `ComputeDistanceByIdx` calculates only the distances between query vectors and the base library vectors at the specified indices. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped `topk` results are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The valid length is `n * dim`, and `dim` must match the dimension specified during initialization.<br>`const int *num`: Number of base library feature vectors to compare for each query. The length is `n`.<br>`const idx_t *indices`: Indices of the base library feature vectors to compare. The number of base library vectors to compare can differ for each query. Store valid vector indices contiguously from front to back and pad the space according to the maximum `num` value. The length of `indices` is `n * max(num)`. If the input is on the host, `indices` is a host pointer. If the input is on the device, `indices` is a device pointer.<br>`MEMORY_TYPE memoryType`: Policy for where the input and output are stored. The default is `MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST`. The available policies are as follows:<br>`MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST`: input on the host and output on the host. `MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE`: input on the device and output on the device. `MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST`: input on the device and output on the host. `MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE`: input on the host and output on the device.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: Distances between the query vectors and the selected base library vectors. Each query records valid distances contiguously from front to back, and the space is padded according to the maximum `num` value. The total length is `n * max(num)`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `num`: User-specified, with length `n`, and each query's `num` value must be in [0, `ntotal`]. `indices`: Each feature index must be in [0, `ntotal` ). For example, if `n = 3` and `num[3] = {1, 3, 5}`, the three queries compare with 1, 3, and 5 base library vectors respectively. Since `max(num) = 5`, the storage space pointed to by `indices` is aligned to 5, and the total size is `3 * 5 * sizeof(idx_t)` bytes, for example, `{{1, 0, 0, 0, 0}, {4, 7, 9, 0, 0}, {1, 3, 4, 7, 9}}`. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. When selecting a `memoryType` storage policy, `queries` and `distances` must be pointers to the corresponding location, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### Finalize API<a name="ZH-CN_TOPIC_0000002482656060"></a>

| API Definition | `void Finalize();` |
| --- | --- |
| Description | Releases feature library management resources. |
| Input | None |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | None |

### GetFeatures API<a name="ZH-CN_TOPIC_0000002484074790"></a>

| API Definition | `APP_ERROR GetFeatures(int n, float *features, const idx_t *indices);` |
| --- | --- |
| Description | Queries the feature vectors with the specified indices for `n` entries. Output is on the host. |
| Input | `int n`: Number of base library vectors to get.<br>`const idx_t *indices`: Indices corresponding to the feature vectors, with length `n`. |
| Output | `float *features`: Feature vectors corresponding to the queried indices. The length is `n * dim`, where `dim` is the vector dimension. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table018415716495"></a>

| API Definition | `APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices);` |
| --- | --- |
| Description | Queries the feature vectors with the specified indices for `n` entries. Output is on the host. |
| Input | `int n`: Number of base library vectors to get.<br>`const idx_t *indices`: Indices corresponding to the feature vectors, with length `n`. |
| Output | `float16_t *features`: Feature vectors corresponding to the queried indices. The length is `n * dim`, where `dim` is the vector dimension. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### GetFeaturesOnDevice API<a name="ZH-CN_TOPIC_0000002516516843"></a>

| API Definition | `APP_ERROR GetFeaturesOnDevice (int n, float16_t *features, const idx_t *indices);` |
| --- | --- |
| Description | Queries the feature vectors with the specified indices for `n` entries. Output is on the Device. |
| Input | `int n`: Number of base library vectors to get.<br>`const idx_t *indices`: Indices corresponding to the feature vectors, with length `n`. |
| Output | `float16_t *features`: Feature vectors corresponding to the queried indices. The length is `n * dim`, where `dim` is the vector dimension. Device-side pointer. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table15312115612410"></a>

| API Definition | `APP_ERROR GetFeaturesOnDevice (int n, float *features, const idx_t *indices);` |
| --- | --- |
| Description | Queries the feature vectors with the specified indices for `n` entries. Output is on the Device. |
| Input | `int n`: Number of base library vectors to get.<br>`const idx_t *indices`: Indices corresponding to the feature vectors, with length `n`. |
| Output | `float *features`: Feature vectors corresponding to the queried indices. The length is `n * dim`, where `dim` is the vector dimension. Device-side pointer. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### GetNTotal API<a name="ZH-CN_TOPIC_0000002514776043"></a>

| API Definition | `int GetNTotal() const;` |
| --- | --- |
| Description | Queries the theoretical maximum number of feature vectors in the current feature library. If the feature vector indices are inserted consecutively, `ntotal` is equal to the number of feature vectors. |
| Input | None |
| Output | `int ntotal`: The theoretical maximum number of feature vectors, that is, the maximum base library index plus 1. |
| Returns | `int`: The theoretical maximum number of feature vectors, that is, the maximum base library index plus 1. |
| Constraints | None |

### Init API<a name="ZH-CN_TOPIC_0000002482736034"></a>

| API Definition | `APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector<int> &deviceList, int64_t resourceSize = -1);` |
| --- | --- |
| Description | Initialization function of `AscendIndexILFlat`. |
| Input | `int dim`: Dimension of the feature vectors managed by `AscendIndexILFlat`.<br>`int capacity`: Maximum base library capacity. The API allocates `capacity * dim * sizeof(fp16)` bytes of memory based on the `capacity` value.<br>`faiss::MetricType metricType`: Feature distance type, including inner product, Euclidean distance, and cosine similarity.<br>`const std::vector<int> &deviceList`: Device-side resource configuration.<br>`int64_t resourceSize`: Device-side preset memory pool size, in bytes. It stores intermediate results during computation and avoids performance fluctuations caused by dynamic memory allocation during computation. The default value is `-1`, which means `128 MB`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `dim` ∈ {32, 64, 128, 256, 384, 512}. `metricType`: `AscendIndexILFlat` currently implements only the inner product distance, so it supports only `faiss::MetricType::METRIC_INNER_PRODUCT`. `capacity`: The maximum memory that the API can allocate for the base library is `12,288,000,000` bytes, and the allowed range of `capacity` is [0, 12000000]. For example, for a base library vector set with 512 dimensions and the FP16 type, the maximum supported `capacity` is 12 million (`12288000000 / (512 * sizeof(fp16))`). For a base library vector set with 256 dimensions and the FP16 type, `capacity` can still be set to at most 12 million, even though the memory limit supports a larger value. Only single-card configuration is supported. Multi-card configuration is not supported yet, and `deviceList.size() == 1` must hold. `resourceSize` can be set to `-1` or any value in [134217728, 4294967296], which is equivalent to `[128 MB, 4096 MB]`. This parameter is determined jointly by the base library size and the search batch size. When the base library contains at least 10 million vectors and the batch size is at least 16, you are advised to set it to `1024 MB`. |

### `operator =` API<a name="ZH-CN_TOPIC_0000002482794858"></a>

| API Definition | `AscendIndexILFlat& operator=(const AscendIndexILFlat &) = delete;` |
| --- | --- |
| Description | Declares this `Index` assignment operator as deleted. Therefore, the type is non-copyable. |
| Input | `const AscendIndexILFlat &`: `AscendIndexILFlat` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### RemoveFeatures API<a name="ZH-CN_TOPIC_0000002482917750"></a>

| API Definition | `APP_ERROR RemoveFeatures(int n, const idx_t *indices);` |
| --- | --- |
| Description | Deletes the feature vectors with the specified indices from the vector library. |
| Input | `int n`: Number of feature vectors to delete.<br>`const idx_t *indices`: Indices of the feature vectors. The length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ), and you can get `ntotal` by calling `GetNTotal`. `n`: The value must be in [0, `capacity` ]. `indices` must be a non-null pointer, and its length must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### Search API<a name="ZH-CN_TOPIC_0000002514896045"></a>

| API Definition | `APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Returns the indices and corresponding distances of the `topk` base library vectors that are closest to the query vectors. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped distances are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`int topk`: Sorts the comparison distances between the query vectors and the base library and returns `topk` results.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: External memory. It stores the cosine distances corresponding to the `topk * n` base library feature vectors that are most similar to the query. The length is `n * topk`.<br>`idx_t *indices`: External memory. It returns the indices corresponding to the `topk` base library vectors that are most similar to the query. The length is `n * topk`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `topk`: The value must be in (0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, and `distances` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table838713119461"></a>

| API Definition | `APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Returns the indices and corresponding distances of the `topk` base library vectors that are closest to the query vectors. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped distances are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float *queries`: Feature vectors to query. The length is `n * dim`, where `dim` is the vector dimension.<br>`int topk`: Sorts the comparison distances between the query vectors and the base library and returns `topk` results.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `float *distances`: External memory. It stores the cosine distances corresponding to the `topk * n` base library feature vectors that are most similar to the query. The length is `n * topk`.<br>`idx_t *indices`: External memory. It returns the indices corresponding to the `topk` base library vectors that are most similar to the query. The length is `n * topk`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `topk`: The value must be in (0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, and `distances` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### SearchByThreshold API<a name="ZH-CN_TOPIC_0000002482656062"></a>

| API Definition | `APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Adds threshold filtering on top of `Search` and returns only the results that meet the threshold condition. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped `topk` results are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float *queries`: Feature vectors to query. The length is `n * dim`.<br>`float threshold`: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by `threshold`.<br>`int topk`: Sorts the comparison distances between the query vectors and the base library and returns `topk` results.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `int *num`: Number of base library vectors that meet the threshold condition for each query. The length is `n`.<br>`idx_t *indices`: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to `topk`. The total length of `indices` is `n * topk`.<br>`float *distances`: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of `indices`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `topk`: The value must be in (0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table910711421721"></a>

| API Definition | `APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);` |
| --- | --- |
| Description | Adds threshold filtering on top of `Search` and returns only the results that meet the threshold condition. If you pass a valid mapping table (`tableLen > 0` and `table` is a non-null pointer), the mapped `topk` results are returned. |
| Input | `int n`: Number of feature vectors to query.<br>`const float16_t *queries`: Feature vectors to query. The length is `n * dim`.<br>`float threshold`: Threshold used for filtering. The API does not restrict the value range. If you pass a mapping table, the API first maps the distances to scores and then filters them by `threshold`.<br>`int topk`: Sorts the comparison distances between the query vectors and the base library and returns `topk` results.<br>`unsigned int tableLen`: Mapping table length. The default value is `0`, which means that mapping is not performed. The currently supported mapping table length is `10000`.<br>`const float *table`: Mapping table pointer. It points to valid mapping values stored in a space of length `tableLen`. The currently supported redundant length is `48`, which means that the space pointed to by `table` has a length of `10048 * sizeof(float)` bytes. |
| Output | `int *num`: Number of base library vectors that meet the threshold condition for each query. The length is `n`.<br>`idx_t* indices`: Indices of the base library vectors that meet the threshold condition. Each query records matching distances from front to back, and then pads the space according to `topk`. The total length of `indices` is `n * topk`.<br>`float *distances`: Distances between the base library vectors that meet the threshold condition and the query vectors. The recording method and length are the same as those of `indices`. |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in (0, `capacity` ]. `topk`: The value must be in (0, 1024]. If you pass `tableLen` and `table` and both satisfy the requirements, the API maps the computed `distance` values.<br>First, it normalizes `distance` to the floating-point value `f1` in [0, 1]. Then it multiplies `f1` by `tableLen` and rounds down to obtain an integer index in [0, `tableLen`]. Next, it uses that integer index as an offset to read the corresponding `score` from the memory space pointed to by `table`, which completes the mapping and stores `score` in `distance`.<br>The index mapping formula can be expressed as `((CosDistance + 1) / 2) * tableLen`. `indices`, `queries`, `distances`, and `num` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

### SetNTotal API<a name="ZH-CN_TOPIC_0000002514776045"></a>

| API Definition | `APP_ERROR SetNTotal(int n);` |
| --- | --- |
| Description | Provides an interface for adjusting the `ntotal` count externally.<br>After base library vectors are added, the `Index` internally updates the `ntotal` value according to the largest inserted index, but it does not record which regions in the range [0, `ntotal` ] are invalid. Therefore, the `RemoveFeatures` operation does not change the `ntotal` value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set `ntotal` manually. This reduces the operator workload within a controllable range and improves interface performance.<br>For example, if 100 vectors are inserted and the base library indices range from 0 to 99, `ntotal = 100`. If you delete the base library entries with indices from 80 to 90, the `ntotal` value inside `Index` remains unchanged and can only be set to a value in [ `ntotal`, `capacity` ]. If you then delete the base library entries with indices from 90 to 99, you can manually set `ntotal` to a value in [80, `capacity` ]. When you set it to `80`, the amount of base library data involved in comparison decreases by 20 vectors. |
| Input | `int n`: Maximum base library index managed by the service side, plus 1. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `n`: The value must be in [0, `capacity` ]. |

### UpdateFeatures API<a name="ZH-CN_TOPIC_0000002516314733"></a>

| API Definition | `APP_ERROR UpdateFeatures (int n, const float16_t *features, const idx_t *indices);` |
| --- | --- |
| Description | Updates `n` feature vectors with the specified indices in the feature library. If a feature vector does not exist at an index, the API adds it. If a feature vector already exists at an index, the API updates it. |
| Input | `int n`: Number of feature vectors to insert.<br>`const float16_t *features`: Feature vectors to insert. The length is `n * dim`, where `dim` is the vector dimension.<br>`const idx_t *indices`: Indices of the feature vectors to insert. The valid length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ). `n`: The value must be in (0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |

<a name="table19567183517113"></a>

| API Definition | `APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);` |
| --- | --- |
| Description | Updates `n` feature vectors with the specified indices in the feature library. If a feature vector does not exist at an index, the API adds it. If a feature vector already exists at an index, the API updates it. |
| Input | `int n`: Number of feature vectors to insert.<br>`const float *features`: Feature vectors to insert. The length is `n * dim`, where `dim` is the vector dimension.<br>`const idx_t *indices`: Indices of the feature vectors to insert. The valid length is `n`. |
| Output | None |
| Returns | `APP_ERROR`: Return status. For details, see the interface return value reference. |
| Constraints | `indices`: Each feature index must be in [0, `ntotal` ). `n`: The value must be in (0, `capacity` ]. `features` and `indices` must be non-null pointers, and their lengths must satisfy the limits. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. |
