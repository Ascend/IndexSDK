# Attribute Filtering-based Retrieval<a name="ZH-CN_TOPIC_0000001482844454"></a>

## AscendIndexTS<a name="ZH-CN_TOPIC_0000001507640105"></a>

### Overview<a name="ZH-CN_TOPIC_0000001507879785"></a>

Spatiotemporal index API class. When you add base library features, you can configure a `FeatureAttr` for each feature. When you run retrieval, you can configure an `AttrFilter` for each batch of query vectors. The filter first screens the entire base library and then compares the vectors that meet the conditions.

The following algorithms are supported:

- Binary feature retrieval (Hamming distance). Before use, manually generate the `BinaryFlat` and `Mask` operators and move them to the corresponding `modelpath` directory.
- `Int8Flat` (cosine distance), `FP16Flat` (IP distance), and `Int8Flat` (L2 distance). Before use, manually generate the `Mask` operator and move it to the corresponding `modelpath` directory.
- Multithreaded concurrent calls are supported. Set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `AddFeature`<a name="ZH-CN_TOPIC_0000001458360182"></a>

| API Definition | `APP_ERROR AddFeature(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const uint8_t *customAttr = nullptr);` |
| --- | --- |
| Description | Adds features. |
| Parameters | `int64_t count`: Number of features to add.<br>`const void *features`: Features to add. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`.<br>`const FeatureAttr *attributes`: Feature attributes to add. For details, see `FeatureAttr`.<br>`const int64_t *labels`: Feature labels to add. Ensure that each label is unique within the `Index` instance.<br>`const uint8_t *customAttr`: User-defined feature attributes to add. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The base library capacity is `1e9`. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `attributes` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `labels` must be `count`, and all elements must be unique and not already exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `customAttr` must be a null pointer or have a length of `count * customAttrLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `customAttrLen` is set in `Init` or `InitWithExtraVal`. |

> [!NOTE]
> `AddFeature` cannot be used together with `AddWithExtraVal`.

### `AddFeatureByIndice`<a name="ZH-CN_TOPIC_0000002411433020"></a>

> [!NOTE]
>
> - `AddFeatureByIndice` cannot be used together with `AddFeature` or `AddWithExtraVal`.
> - After you use `AddFeatureByIndice` to add base library features by position, you cannot use APIs such as `GetExtraValAttrByLabel` that depend on labels. `AddFeatureByIndice` and `GetFeatureByIndice` must be used as a pair.

| API Definition | `APP_ERROR AddFeatureByIndice(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *indices, const ExtraValAttr *extraVal = nullptr, const uint8_t *customAttr = nullptr);` |
| --- | --- |
| Description | Adds base library features by position. This API currently supports only `FlatIP` and `Int8Flat` (cosine distance). |
| Parameters | `int64_t count`: Number of features to add.<br>`const void *features`: Features to add. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`.<br>`const FeatureAttr *attributes`: Feature attributes to add.<br>`const int64_t *indices`: Positions of the features in the base library.<br>`const ExtraValAttr *extraVal`: Additional feature attributes to add.<br>`const uint8_t *customAttr`: User-defined feature attributes to add. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The base library capacity is `1e9`. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `attributes` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `indices` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The values must be strictly increasing and non-negative. If a value is smaller than the number of features in the base library, it indicates replacement. If a value is greater than or equal to the number of features in the base library, it indicates addition, and the values must be consecutive. `extraVal` must be a null pointer or have a length of `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. A null pointer means that no additional attributes need to be added. `customAttr` must be a null pointer or have a length of `count * customAttrLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. A null pointer means that no custom attributes need to be added. |

### `AddWithExtraVal`<a name="ZH-CN_TOPIC_0000001976650872"></a>

| API Definition | `APP_ERROR AddWithExtraVal(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const ExtraValAttr *extraVal, const uint8_t *customAttr = nullptr);` |
| --- | --- |
| Description | Adds features with additional attributes. |
| Parameters | `int64_t count`: Number of features to add.<br>`const void *features`: Features to add. The Hamming distance uses `uint8_t` data, and `Int8Flat` uses `int8_t`.<br>`const FeatureAttr *attributes`: Feature attributes to add. For details, see `FeatureAttr`.<br>`const int64_t *labels`: Feature labels to add. Ensure that each label is unique within the `Index` instance.<br>`const ExtraValAttr *extraVal`: Additional feature attributes to add. For details, see `ExtraValAttr`.<br>`const uint8_t *customAttr`: User-defined feature attributes to add. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The base library capacity is `1e9`. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `attributes` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `labels` must be `count`, and all elements must be unique and not already exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `extraVal` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `customAttr` must be a null pointer or have a length of `count * customAttrLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `customAttrLen` is set in `Init` or `InitWithExtraVal`. |

### `AscendIndexTS`<a name="ZH-CN_TOPIC_0000001458200394"></a>

| API Definition | `AscendIndexTS() = default;` |
| --- | --- |
| Description | Constructor of `AscendIndexTS`. |
| Parameters | None |
| Output | None |
| Returns | None |
| Constraints | None |

| API Definition | `AscendIndexTS(const AscendIndexTS &) = delete;` |
| --- | --- |
| Description | Copy constructor of `AscendIndexTS`. |
| Parameters | `const AscendIndexTS &`: `AscendIndexTS` object. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendIndexTS`<a name="ZH-CN_TOPIC_0000001507760865"></a>

| API Definition | `virtual ~AscendIndexTS() = default;` |
| --- | --- |
| Description | Destructor of `AscendIndexTS`. It destroys the feature management object. |
| Parameters | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `DeleteFeatureByLabel`<a name="ZH-CN_TOPIC_0000001458200398"></a>

| API Definition | `APP_ERROR DeleteFeatureByLabel(int64_t count, const int64_t *labels);` |
| --- | --- |
| Description | Deletes features by label. |
| Parameters | `int64_t count`: Number of features to delete.<br>`const int64_t *labels`: Feature labels. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `labels` must be `count`, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `DeleteFeatureByToken`<a id="ZH-CN_TOPIC_0000001458680018"></a>

| API Definition | `APP_ERROR DeleteFeatureByToken(int64_t count, const uint32_t *tokens);` |
| --- | --- |
| Description | Deletes features by token. |
| Parameters | `int64_t count`: Number of features to delete.<br>`const uint32_t *tokens`: Feature tokens. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `tokens` must be `count`, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `FastDeleteFeatureByIndice`<a name="ZH-CN_TOPIC_0000002445152089"></a>

| API Definition | `APP_ERROR FastDeleteFeatureByIndice(int64_t count, const int64_t *indices);` |
| --- | --- |
| Description | Quickly deletes features by position. |
| Parameters | `int64_t count`: Number of features to delete.<br>`const int64_t *indices`: Positions of the features in the base library. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `indices` must be `count`, and all values must be unique, non-negative, and smaller than the number of features in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `FastDeleteFeatureByRange`<a name="ZH-CN_TOPIC_0000002445960745"></a>

| API Definition | `APP_ERROR FastDeleteFeatureByRange(int64_t start, int64_t count);` |
| --- | --- |
| Description | Quickly deletes `count` base library features starting from `start`. This API supports only the additional similarity scenarios of `TSFlatIP` and `TSInt8FlatCos`. |
| Parameters | `int64_t start`: Start position of the features to delete.<br>`int64_t count`: Number of features to delete. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `start` must be greater than or equal to 0 and smaller than the number of features in the base library. `count` must be greater than 0 and less than or equal to the number of features in the base library. The sum of `start` and `count` must be less than or equal to the number of features in the base library. |

### `GetBaseByRange`<a name="ZH-CN_TOPIC_0000001818301380"></a>

| API Definition | `APP_ERROR GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes);` |
| --- | --- |
| Description | Queries the base library by range. |
| Parameters | `uint32_t offset`: Initial offset for retrieving base library features.<br>`uint32_t num`: Number of features. |
| Output | `int64_t *labels`: Feature labels.<br>`void *features`: Features. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`.<br>`FeatureAttr *attributes`: Feature attributes. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `0 < offset <= 8.0e8`. `0 < num <= 8.0e8`. `offset + num <= ntotal`. The length of `labels` must be `num`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `features` must be `num * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `attributes` must be `num`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetBaseByRangeWithExtraVal`<a name="ZH-CN_TOPIC_0000001976495686"></a>

| API Definition | `APP_ERROR GetBaseByRangeWithExtraVal(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes, ExtraValAttr *extraVal) const;` |
| --- | --- |
| Description | Queries the base library with additional attributes by range. |
| Parameters | `uint32_t offset`: Initial offset for retrieving base library features.<br>`uint32_t num`: Number of features. |
| Output | `int64_t *labels`: Feature labels.<br>`void *features`: Features. The Hamming distance uses `uint8_t` data, and `Int8Flat` uses `int8_t`.<br>`FeatureAttr *attributes`: Feature attributes.<br>`ExtraValAttr *extraVal`: Additional attributes. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `0 <= offset < 8.0e8`. `0 < num <= 8.0e8`. `offset + num <= ntotal`. The length of `labels` must be `num`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `features` must be `num * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `attributes` must be `num`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `extraVal` must be `num`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetBaseMask`<a name="ZH-CN_TOPIC_0000002445112157"></a>

| API Definition | `APP_ERROR GetBaseMask(int64_t count, uint8_t *mask);` |
| --- | --- |
| Description | Obtains the flag that indicates whether the base library has been quickly deleted. If a bit is 0, the base library entry at that position has been deleted and is invalid. |
| Parameters | `int64_t count`: Valid length of the `mask` array. |
| Output | `uint8_t *mask`: Array that marks whether the base library entry has been deleted. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, ceil(`ntotal`/8)]. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, `ntotal` is the number of features in the base library. The length of `mask` must be greater than or equal to `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetCustomAttrByBlockId`<a name="ZH-CN_TOPIC_0000001736682593"></a>

| API Definition | `APP_ERROR GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const;` |
| --- | --- |
| Description | Obtains the custom attributes of the specified `blockId`. |
| Parameters | `uint32_t blockId`: `blockId` to retrieve.<br>`uint8_t *&customAttr`: User-defined feature attributes on the device side. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | The length of `customAttr` must be `customAttrBlockSize * customAttrLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `customAttrBlockSize` and `customAttrLen` are set in `Init` or `InitWithExtraVal`. |

### `GetExtraValAttrByLabel`<a name="ZH-CN_TOPIC_0000001976655414"></a>

| API Definition | `APP_ERROR GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const;` |
| --- | --- |
| Description | Obtains the additional attributes of the features with the specified labels. |
| Parameters | `int64_t count`: Number of features to retrieve.<br>`const int64_t *labels`: Feature labels. |
| Output | `ExtraValAttr *extraVal`: Additional attributes. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `labels` must be `count`, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. If the input `labels` do not exist in the base library, the `val` field in the returned additional attributes is `INT16_MIN`. The length of `extraVal` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetFeatureAttrByLabel`<a name="ZH-CN_TOPIC_0000001594544301"></a>

| API Definition | `APP_ERROR GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const;` |
| --- | --- |
| Description | Obtains the attributes of the features with the specified labels. |
| Parameters | `int64_t count`: Number of features to retrieve.<br>`const int64_t *labels`: Feature labels. |
| Output | `FeatureAttr *attributes`: Feature attributes. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `labels` must be `count`, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. If the input `labels` do not exist in the base library, the returned feature attributes contain `time = INT32_MIN` and `tokenId = UINT32_MAX`. The length of `attributes` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetFeatureByIndice`<a name="ZH-CN_TOPIC_0000002411592888"></a>

| API Definition | `APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels = nullptr, void *features = nullptr, FeatureAttr *attributes = nullptr, ExtraValAttr *extraVal = nullptr) const;` |
| --- | --- |
| Description | Obtains base library features by position. |
| Parameters | `int64_t count`: Number of features to retrieve.<br>`const int64_t *indices`: Positions of the features in the base library. |
| Output | `int64_t *labels`: Labels of the features to retrieve.<br>`void *features`: Feature vectors to retrieve.<br>`FeatureAttr *attributes`: Spatiotemporal attributes of the features to retrieve.<br>`ExtraValAttr *extraVal`: Additional attributes of the features to retrieve. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `indices` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The values must be greater than or equal to 0 and less than the number of features in the base library. When `labels` is `nullptr`, no labels are retrieved. Otherwise, the length must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `features` is `nullptr`, no features are retrieved. Otherwise, the length must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `attributes` is `nullptr`, no attributes are retrieved. Otherwise, the length must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `extraVal` is `nullptr`, no additional attributes are retrieved. Otherwise, the length must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetFeatureByLabel`<a name="ZH-CN_TOPIC_0000001507879789"></a>

| API Definition | `APP_ERROR GetFeatureByLabel(int64_t count, const int64_t *labels, void *features);` |
| --- | --- |
| Description | Retrieves the features with the specified labels. |
| Parameters | `int64_t count`: Number of features to retrieve.<br>`const int64_t *labels`: Feature labels. |
| Output | `void *features`: Features retrieved by the specified labels. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 1e6]. The length of `labels` must be `count`, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `GetFeatureNum`<a name="ZH-CN_TOPIC_0000001544946953"></a>

| API Definition | `APP_ERROR GetFeatureNum(int64_t *totalNum);` |
| --- | --- |
| Description | Obtains the number of features in this `Index` instance. |
| Parameters | None |
| Output | `int64_t *totalNum`: Number of features in the base library. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | None |

### `Init`<a id="ZH-CN_TOPIC_0000001458680014"></a>

| API Definition | `APP_ERROR Init(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, AlgorithmType algType = AlgorithmType::FLAT_COS_INT8, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits<uint64_t>::max());` |
| --- | --- |
| Description | Initializes the instance. |
| Parameters | `uint32_t deviceId`: Device ID used by the `Index`.<br>`uint32_t dim`: Dimension of the base library vectors.<br>`uint32_t tokenNum`: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated `Mask` operator.<br>`AlgorithmType algType`: Distance comparison algorithm used by the backend. The default value is `AlgorithmType::FLAT_COS_INT8`. For supported algorithms, see the following list.<br> `AlgorithmType::FLAT_HAMMING`: Binary feature retrieval (Hamming distance). `AlgorithmType::FLAT_COS_INT8`: `Int8Flat` (cosine distance). `AlgorithmType::FLAT_L2_INT8`: `Int8Flat` (L2 distance). `AlgorithmType::FLAT_IP_FP16`: `FP16Flat` (IP distance). `AlgorithmType::FLAT_HPP_COS_INT8`: `Int8Flat` (cosine distance).<br>`MemoryStrategy memoryStrategy`: Memory strategy used by the backend. The default value is `MemoryStrategy::PURE_DEVICE_MEMORY`. Supported strategies are listed below. `MemoryStrategy::PURE_DEVICE_MEMORY`: Pure device memory strategy. `MemoryStrategy::HETERO_MEMORY`: Heterogeneous memory strategy. `MemoryStrategy::HPP`: HPP heterogeneous memory strategy.<br>`customAttrLen`: Length of custom attributes.<br>`customAttrBlockSize`: Block size of custom attributes.<br>`maxFeatureRowCount`: Maximum number of vectors in the base library. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | Call this API immediately after the constructor. `deviceId` must be a valid device ID in the range [0, 1024]. `tokenNum` must be in the range (0, 3e5]. For binary feature retrieval (Hamming distance), `dim` must be one of {256, 512, 1024}. For the `Int8Flat` algorithm (cosine distance or L2 distance), `dim` must be one of {64, 128, 256, 384, 512, 768, 1024}. For the `FP16Flat` algorithm (IP distance), `dim` must be one of {64, 128, 256, 384, 512, 768, 1024}. `memoryStrategy::HETERO_MEMORY` currently supports only `AlgorithmType::FLAT_COS_INT8`. `customAttrLen` must be in the range [0, 32]. The default value is 0. A value of 0 means that no custom attributes exist. `customAttrBlockSize` must be in the range [0, 262144*64] and must be an integer multiple of 1024*256. The default value is 0. A value of 0 means that no custom attributes exist. `maxFeatureRowCount` must be in the range [262144 \* 64, 262144 \* 550 \* 3] and must be an integer multiple of 256. The default value is the maximum value of `uint64`. This parameter is valid only when `memoryStrategy` is set to `MemoryStrategy::HPP`. When `memoryStrategy` is set to `MemoryStrategy::HPP`, the available Host memory must be at least 250 GB, the number of free physical CPU cores must be at least 15, and only 256-dimensional vector retrieval is supported. |

### `InitWithExtraVal`<a id="ZH-CN_TOPIC_0000002013206217"></a>

| API Definition | `APP_ERROR InitWithExtraVal(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, AlgorithmType algType = AlgorithmType::FLAT_HAMMING, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits<uint64_t>::max());` |
| --- | --- |
| Description | Initializes an instance with additional attributes. |
| Parameters | `uint32_t deviceId`: Device ID used by the `Index`.<br>`uint32_t dim`: Dimension of the base library vectors.<br>`uint32_t tokenNum`: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated `Mask` operator.<br>`uint64_t resources`: Shared memory size.<br>`AlgorithmType algType`: Distance comparison algorithm used by the backend. The default value is `AlgorithmType::FLAT_HAMMING`. Supported algorithms are listed below. `AlgorithmType::FLAT_HAMMING`: Binary feature retrieval (Hamming distance). `AlgorithmType::FLAT_COS_INT8`: `Int8Flat` (cosine distance).<br>`MemoryStrategy memoryStrategy`: Memory strategy used by the backend. The default value is `MemoryStrategy::PURE_DEVICE_MEMORY`. Supported strategies are listed below. `MemoryStrategy::PURE_DEVICE_MEMORY`: Pure device memory strategy. `MemoryStrategy::HETERO_MEMORY`: Heterogeneous memory strategy.<br>`customAttrLen`: Length of custom attributes.<br>`customAttrBlockSize`: Block size of custom attributes.<br>`maxFeatureRowCount`: Maximum number of vectors in the base library. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | Call this API immediately after the constructor. `deviceId` must be a valid device ID in the range [0, 1024]. `tokenNum` must be in the range (0, 3e5]. `resources` must be in the range [1 \* 1024 \* 1024 \* 1024, 32 \* 1024 \* 1024 \* 1024]. When you use additional attributes, 4 GB is recommended. For binary feature retrieval (Hamming distance), `dim` must be one of {256, 512, 1024}. For the `Int8Flat` algorithm (cosine distance), `dim` must be one of {64, 128, 256, 384, 512, 768, 1024}. `customAttrLen` must be in the range [0, 32]. The default value is 0. A value of 0 means that no custom attributes exist. `customAttrBlockSize` must be in the range [0, 262144 \* 64] and must be an integer multiple of 1024*256. The default value is 0. A value of 0 means that no custom attributes exist. `maxFeatureRowCount` does not support HPP when additional attributes are used, and the default value is the maximum value of `uint64`. |

### `InitWithQuantify`<a name="ZH-CN_TOPIC_0000002458673509"></a>

| API Definition | `APP_ERROR InitWithQuantify(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, const float *scale, AlgorithmType algType = AlgorithmType::FLAT_IP_FP16, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0);` |
| --- | --- |
| Description | Initializes the vectorized base library. |
| Parameters | `uint32_t deviceId`: Device ID used by the `Index`.<br>`uint32_t dim`: Dimension of the base library vectors.<br>`uint32_t tokenNum`: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated `Mask` operator.<br>`uint64_t resources`: Shared memory size.<br>`const float *scale`: Scaling factor for base library vectorization. After the scaling factor is multiplied by the base library, the result is converted to the `int8_t` type.<br>`AlgorithmType algType`: Distance comparison algorithm used by the backend. The default value is `AlgorithmType::FLAT_IP_FP16`, which means `FP16Flat` (IP distance). Currently, only `AlgorithmType::FLAT_IP_FP16` is supported.<br>`uint32_t customAttrLen`: Length of custom attributes.<br>`uint32_t customAttrBlockSize`: Block size of custom attributes. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | Call this API immediately after the constructor. `deviceId` must be a valid device ID in the range [0, 1024]. `tokenNum` must be in the range (0, 3e5]. `resources` must be greater than 0 and less than or equal to 4 \* 1024 \* 1024 \* 1024. The `scale` array is used for division during dequantization and must not be close to 0. The absolute value of each factor in `scale` must be greater than or equal to 1e-6f. For the `FP16Flat` algorithm (IP distance), `dim` must be one of {64, 128, 256, 384, 512, 768, 1024}. Only the non-shared mode of the `FP16Flat` algorithm (IP distance) is supported. This API must be used together with `AddFeatureByIndice`. `customAttrLen` must be in the range [0, 32]. The default value is 0. A value of 0 means that no custom attributes exist. `customAttrBlockSize` must be in the range [0, 262144 * 64] and must be an integer multiple of 1024 \* 256. The default value is 0. A value of 0 means that no custom attributes exist. |

### `operator =`<a name="ZH-CN_TOPIC_0000001507959881"></a>

| API Definition | `AscendIndexTS &operator=(const AscendIndexTS &) = delete;` |
| --- | --- |
| Description | Declares the assignment operator for this `Index` as deleted, which means that the type is non-copyable. |
| Parameters | `const AscendIndexTS &`: Constant `AscendIndexTS`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `Search`<a name="ZH-CN_TOPIC_0000001507640109"></a>

| API Definition | `APP_ERROR Search(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter = true);` |
| --- | --- |
| Description | Calculates the distance between the input features and the base library vectors filtered by `AttrFilter`, sorts the distances by TopK, and returns the corresponding distances and indices. |
| Parameters | `uint32_t count`: Number of features to compare.<br>`const void *features`: Features to compare. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`.<br>`const AttrFilter *attrFilter`: Attribute filter information. For details, see `AttrFilter`.<br>`bool shareAttrFilter`: Whether different queries share the same mask.<br>`uint32_t topk`: TopK size to keep after cosine distance calculation.<br>`bool enableTimeFilter`: Time-stamp attribute filter switch. The default value is `true`. When `enableTimeFilter = false`, time-stamp attribute filtering is disabled. |
| Output | `int64_t *labels`: Labels of the TopK features.<br>`float *distances`: Distances of the TopK features.<br>`uint32_t *validNums`: Number of valid results obtained after each query vector is compared. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 10240]. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `shareAttrFilter` is `true`, the length of `attrFilter` must be 1. When `shareAttrFilter` is `false`, the length must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `topk` must be in the range [1, 100000]. The length of `labels` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `distances` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `validNums` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `SearchWithExtraMask`<a name="ZH-CN_TOPIC_0000001494506850"></a>

| API Definition | `APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);` |
| --- | --- |
| Description | Calculates the distance between the input features and the base library vectors filtered by `AttrFilter` and an external `Mask`, sorts the distances by TopK, and returns the corresponding distances and indices. |
| Parameters | `uint32_t count`: Number of features to compare.<br>`const void *features`: Features to compare. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`.<br>`const AttrFilter *attrFilter`: Attribute filter information. For details, see `AttrFilter`.<br>`bool shareAttrFilter`: Whether the same query shares one `Mask`.<br>`uint32_t topk`: TopK size to keep after cosine distance calculation.<br>`const uint8_t *extraMask`: Additional filter `Mask` provided from outside. The value is expressed in bits, where 0 and 1 indicate filtering or selecting the feature respectively.<br>`uint64_t extraMaskLenEachQuery`: Length of the external `Mask`, in bytes.<br>`bool extraMaskIsAtDevice`: Whether the external `Mask` already exists on the device side.<br>`bool enableTimeFilter`: Time-stamp attribute filter switch. The default value is `true`. When `enableTimeFilter = false`, time-stamp attribute filtering is disabled. |
| Output | `int64_t *labels`: Labels of the TopK features.<br>`float *distances`: Distances of the TopK features.<br>`uint32_t *validNums`: Number of valid results obtained after each query vector is compared. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 10240]. `topk` must be in the range [1, 100000]. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `shareAttrFilter` is `true`, the length of `attrFilter` must be 1. When `shareAttrFilter` is `false`, the length must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `distances` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `validNums` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `labels` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `shareAttrFilter` is `true`, the length of `extraMask` must be `extraMaskLenEachQuery`. When `shareAttrFilter` is `false`, the length of `extraMask` must be `count * extraMaskLenEachQuery`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `SearchWithExtraMask` with Extra Similarity<a name="ZH-CN_TOPIC_0000002373091106"></a>

| API Definition | `APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, const uint16_t *extraScore, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);` |
| --- | --- |
| Description | Calculates the distance between the input features and the base library vectors filtered by `AttrFilter` and an external `Mask`, sorts the distances by TopK, and returns the corresponding distances and indices. |
| Parameters | `uint32_t count`: Number of features to compare.<br>`const void *features`: Features to compare. The Hamming distance uses `uint8_t` data, `Int8Flat` uses `int8_t`, and `FP16Flat` uses `float`.<br>`const AttrFilter *attrFilter`: Attribute filter information. For details, see `AttrFilter`.<br>`bool shareAttrFilter`: Whether the same query shares one `Mask`.<br>`uint32_t topk`: TopK size to keep after cosine distance calculation.<br>`const uint8_t *extraMask`: Additional filter `Mask` provided from outside. The value is expressed in bits, where 0 and 1 indicate filtering or selecting the feature respectively.<br>`uint64_t extraMaskLenEachQuery`: Length of the external `Mask`, in bytes.<br>`bool extraMaskIsAtDevice`: Whether the external `Mask` already exists on the device side.<br>`const uint16_t *extraScore`: Additional similarity provided by the user. The length is `count * totalPad`, where `totalPad` is the base library length aligned to 16 bytes.<br>`bool enableTimeFilter`: Time-stamp attribute filter switch. The default value is `true`. When `enableTimeFilter = false`, time-stamp attribute filtering is disabled. |
| Output | `int64_t *labels`: Labels of the TopK features. If the base library is added by using `AddFeatureByIndice`, the output here is the base library position (`indices`).<br>`float *distances`: Distances of the TopK features.<br>`uint32_t *validNums`: Number of valid results obtained after each query vector is compared. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 10240]. `topk` must be in the range [1, 100000]. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `shareAttrFilter` is `true`, the length of `attrFilter` must be 1. When `shareAttrFilter` is `false`, the length must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `distances` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `validNums` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `labels` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `shareAttrFilter` is `true`, the length of `extraMask` must be `extraMaskLenEachQuery`. When `shareAttrFilter` is `false`, the length of `extraMask` must be `count * extraMaskLenEachQuery`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `extraScore` must be `count * totalPad`, where `totalPad` is the base library length aligned to 16 bytes. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. It actually corresponds to the `float16_t` type, and its values range from `-1.0` to `1.0`. It is currently effective only for non-shared masks in `Int8FlatCos` and `FlatIP`. Otherwise, `extraScore` does not take part in the calculation. |

### `SearchWithExtraVal`<a name="ZH-CN_TOPIC_0000002013215285"></a>

| API Definition | `APP_ERROR SearchWithExtraVal(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, const ExtraValFilter *extraValFilter, bool enableTimeFilter = true);` |
| --- | --- |
| Description | Calculates the distance between the input features and the base library vectors filtered by `AttrFilter` and `ExtraValFilter`, sorts the distances by TopK, and returns the corresponding distances and indices. |
| Parameters | `uint32_t count`: Number of features to compare.<br>`const void *features`: Features to compare. The Hamming distance uses `uint8_t` data, and `Int8cos` uses `int8_t`. Currently, only `int8cos` is supported, including heterogeneous memory scenarios, together with the Hamming distance.<br>`const AttrFilter *attrFilter`: Attribute filter information. For details, see `AttrFilter`.<br>`bool shareAttrFilter`: Additional attributes currently support only `false`. Different queries do not share the same mask.<br>`uint32_t topk`: TopK size to keep after cosine distance calculation.<br>`const ExtraValFilter *extraValFilter`: Additional attribute filter information. For details, see `ExtraValFilter`.<br>`bool enableTimeFilter`: Time-stamp attribute filter switch. The default value is `true`. When `enableTimeFilter = false`, time-stamp attribute filtering is disabled. |
| Output | `uint32_t *validNums`: Number of valid results obtained after each query vector is compared.<br>`int64_t *labels`: Labels of the TopK features.<br>`float *distances`: Distances of the TopK features. |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | `count` must be in the range [1, 10240]. The length of `features` must be `count * dim`, where `dim` is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When `shareAttrFilter` is `true`, the length of `attrFilter` must be 1. When `shareAttrFilter` is `false`, the length must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `topk` must be in the range [1, 100000]. The length of `labels` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `distances` must be `count * topk`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of `validNums` must be `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. `extraValFilter` must be a null pointer or have a length of `count`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

> [!NOTE]
>
> `SearchWithExtraVal` cannot be used together with `Search`.

### `SetHeteroParam`<a name="ZH-CN_TOPIC_0000001630850578"></a>

| API Definition | `APP_ERROR SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);` |
| --- | --- |
| Description | Sets the parameters of the heterogeneous storage strategy. |
| Parameters | `size_t deviceCapacity`: Base library capacity stored on the device side when the heterogeneous memory strategy is used, in bytes.<br>`size_t deviceBuffer`: Cache capacity on the device side when the heterogeneous memory strategy is used, in bytes.<br>`size_t hostCapacity`: Base library capacity stored on the Host side when the heterogeneous memory strategy is used, in bytes. |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | Use this API after you set the memory strategy to `MemoryStrategy::HETERO_MEMORY` in the `Init` API. The minimum value of `deviceCapacity` is `1G`, and the maximum value is the actual remaining device memory. The minimum value of `deviceBuffer` is `2 * 262144 * dim`, and the maximum value is `8G`. Set it according to the actual remaining device memory. `deviceCapacity + deviceBuffer` must be smaller than the actual remaining device memory on the device. The value range of `hostCapacity` is `[1G, 512G]`. Configure it according to the amount of actual memory that can be allocated on the Host side. |

### `SetSaveHostMemory`<a name="ZH-CN_TOPIC_0000002106649489"></a>

| API Definition | `APP_ERROR SetSaveHostMemory();` |
| --- | --- |
| Description | Sets the host memory saving mode. This mode is disabled by default. |
| Parameters | None |
| Output | None |
| Returns | `APP_ERROR`: Operation status. For details, see the API return value reference. |
| Constraints | Use this API after the `Init` API when the base library size is 0. This API can save host memory, but it reduces the performance of delete-type and retrieve-type APIs. When you use this mode, you cannot use the `DeleteFeatureByToken` API. This API supports only the Hamming distance. |

## `AttrFilter`<a id="ZH-CN_TOPIC_0000001458687398"></a>

### Overview<a name="ZH-CN_TOPIC_0000001507967265"></a>

Feature attribute filter. This structure must be used together with an `AscendIndexTS` instance and acts as an input parameter during feature retrieval.

All query vectors in a retrieval call share the same filter. The filter matches the attributes of each base library feature. The comparable information includes time and token ID.

Matched base library features participate in the retrieval process that follows, including vector distance comparison and TopK sorting.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device. The retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `timesEnd`<a name="ZH-CN_TOPIC_0000001458367566"></a>

`int32_t`: End time of the filter interval.

### `timesStart`<a name="ZH-CN_TOPIC_0000001507647493"></a>

`int32_t`: Start time of the filter interval.

### `tokenBitSet`<a name="ZH-CN_TOPIC_0000001507887177"></a>

`uint8_t*`: List of feature token IDs. Each `uint8_t` member records token information bit by bit from low-order bits to high-order bits. 1 indicates selected, and 0 indicates that the token is not selected.

For example, if a filter token list contains two non-zero `uint8_t` members, `[7, 15, 0, 0, ..., 0]`, and the binary representations of the non-zero members are 00000111 and 00001111, the valid token IDs they represent are 0, 1, 2, 8, 9, 10, and 11.

> [!NOTE]
> The length of `tokenBitSet` should be `tokenBitSetLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.

### `tokenBitSetLen`<a name="ZH-CN_TOPIC_0000001458687402"></a>

`uint32_t`: Length of the `tokenBitSet` field in `AttrFilter`.

## `ExtraValAttr`<a id="ZH-CN_TOPIC_0000002013198657"></a>

### Overview<a name="ZH-CN_TOPIC_0000002013039153"></a>

Additional attribute information. It is added together with the feature vector when the feature is stored. This structure must be used together with an `AscendIndexTS` instance.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device.

### `val`<a name="ZH-CN_TOPIC_0000001976479160"></a>

`int16_t`: Records the additional attribute information of the current feature. The binary representation uses 1 to indicate `yes` and 0 to indicate `no`.

## `ExtraValFilter`<a id="ZH-CN_TOPIC_0000002013200765"></a>

### Overview<a name="ZH-CN_TOPIC_0000001976640904"></a>

Additional attribute filter. This structure must be used together with an `AscendIndexTS` instance and acts as an input parameter during feature retrieval.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device.

### `filterVal`<a name="ZH-CN_TOPIC_0000001976481180"></a>

`int16_t`: Additional attributes to query. The binary representation uses 1 to indicate that the additional attribute is retained and 0 to indicate that it is filtered out.

### `matchVal`<a name="ZH-CN_TOPIC_0000002013041289"></a>

`int16_t`: Additional attribute query mode. Two modes are supported, mode 0 and mode 1.

- For mode 0, the matching condition is `ExtraValAttr::val & ExtraValFilter::filterVal == ExtraValFilter::filterVal`.
- For mode 1, the matching condition is `ExtraValAttr::val & ExtraValFilter::filterVal > 0`.

## `FeatureAttr`<a id="ZH-CN_TOPIC_0000001507967381"></a>

### Overview<a name="ZH-CN_TOPIC_0000001458367674"></a>

Feature attribute information. It is added together with the feature vector when the feature is stored. This structure must be used together with an `AscendIndexTS` instance.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device.

### `time`<a name="ZH-CN_TOPIC_0000001507647601"></a>

`int32_t`: Records the time information of the current feature as a time stamp in seconds.

> [!NOTE]
> Due to Ascend hardware limitations, only `int32` type data can be processed. Therefore, you need to ensure that the current time stamp does not exceed the maximum value of `int32`. In actual operations, subtract a fixed historical time stamp from the current time stamp before you store it.

### `tokenId`<a name="ZH-CN_TOPIC_0000001507887269"></a>

`uint32_t`: Feature token ID. One token ID corresponds to multiple features, and one feature corresponds to one token ID. The value must be smaller than `tokenNum` passed when the user initializes `AscendIndexTS`.
