# Other Functions<a name="ZH-CN_TOPIC_0000001482684458"></a>

## `IReduction`<a name="ZH-CN_TOPIC_0000001456694992"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506615161"></a>

`IReduction` is the unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports the `PCAR` and `NN` dimensionality reduction algorithms.

### `CreateReduction`<a name="ZH-CN_TOPIC_0000001456695108"></a>

| API Definition | ```IReduction *CreateReduction(std::string typeName, const ReductionConfig &config);``` |
| --- | --- |
| Description | Creates a specific dimensionality reduction algorithm. |
| Input | `std::string typeName`: Dimensionality reduction algorithm parameter. Valid values are `{"NN", "PCAR"}`.<br>`ReductionConfig &config`: Dimensionality reduction configuration. |
| Output | None |
| Returns | `IReduction *CreateReduction`: Created dimensionality reduction instance. |
| Constraints | Currently, only the `NN` and `PCAR` dimensionality reduction parameters are supported. Using any other parameter causes an exception.<br>After you finish using this instance, remember to `delete` this pointer to release the corresponding memory. |

### `reduce`<a name="ZH-CN_TOPIC_0000001456375280"></a>

| API Definition | ```virtual void reduce(idx_t n, const float *x, float *res) const = 0;``` |
| --- | --- |
| Description | Dimensionality reduction interface. This function does not provide a concrete implementation. |
| Input | `idx_t n`: Number of inputs for inference.<br>`const float *x`: Feature vectors for inference. |
| Output | `float *res`: Feature-vector results from inference. |
| Returns | None |
| Constraints | The value of `n` must be in the range 0 < `n` < 1e9. Pointer `x` must be non-null and its length must be `dimIn * n`. Pointer `res` must be non-null and its length must be `dimOut * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `ReductionConfig`<a name="ZH-CN_TOPIC_0000001456375264"></a>

| Member | Type | Description |
|--|--|--|
| dimIn | int | Input feature dimension, that is, the dimension before reduction. `PCAR` requires this parameter. |
| dimOut | int | Output feature dimension, that is, the dimension after reduction. `PCAR` requires this parameter. |
| eigenPower | float | Power of the singular values. `PCAR` requires this parameter. |
| randomRotation | bool | Whether to perform random rotation. `PCAR` requires this parameter. |
| deviceList | std::vector\<int> | Device-side resource configuration. `NN` requires this parameter. |
| model | const char * | Neural network dimensionality reduction model. `NN` requires this parameter. |
| modelSize | uint64_t | Model size. `NN` requires this parameter. |

<a name="table7235918388"></a>

| API Definition | ```inline ReductionConfig(int dimIn, int dimOut, float eigenPower, bool randomRotation);``` |
| --- | --- |
| Description | Default constructor of `ReductionConfig`. Use this function when you use `PCAR` dimensionality reduction. |
| Input | `int dimIn`: Input feature dimension, that is, the dimension before reduction. `PCAR` requires this parameter.<br>`int dimOut`: Output feature dimension, that is, the dimension after reduction. `PCAR` requires this parameter.<br>`float eigenPower`: Power of the singular values. `PCAR` requires this parameter.<br>`bool randomRotation`: Whether to perform random rotation. `PCAR` requires this parameter. |
| Output | None |
| Returns | None |
| Constraints | When you use different dimensionality reduction algorithms, configure the corresponding parameters, and ensure that the dimension after reduction satisfies the dimension limit of the downstream index that uses the reduced data. When you use `PCAR` dimensionality reduction, ensure that `dimOut` > 0 and `dimIn` >= `dimOut`. The range of `eigenPower` is [-0.5, 0]. |

<a name="table2034112619"></a>

| API Definition | ```inline ReductionConfig(std::vector<int> deviceList, const char *model, uint64_t modelSize);``` |
| --- | --- |
| Description | Default constructor of `ReductionConfig`. Use this function when you use `NN` dimensionality reduction. |
| Input | `std::vector<int> deviceList`: Device-side resource configuration.<br>`const char *model`: Neural network dimensionality reduction model.<br>`uint64_t modelSize`: Model size. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `deviceList` is (0, 32]. <br>When you use different dimensionality reduction algorithms, configure the corresponding parameters, and ensure that the dimension after reduction satisfies the dimension limit of the downstream index that uses the reduced data. <br>`model` must be a valid, effective memory pointer to a deep neural network dimensionality reduction model, and its size must be `modelSize`. The valid range of `modelSize` is (0, 128 MB]. Parameter mismatches may cause model instantiation or inference to fail. Invalid models may harm the system. Ensure that the model source is valid and effective. <br>`dimsIn` ∈ {64, 128, 256, 384, 512, 768, 1024}. <br>`dimsOut` ∈ {32, 64, 96, 128, 256}. <br>`batches` ∈ {1, 2, 4, 8, 16, 32, 64, 128}. |

### `~IReduction`<a name="ZH-CN_TOPIC_0000001714244661"></a>

<a name="table7235918388"></a>

| API Definition | ```virtual ~IReduction() = default;``` |
| --- | --- |
| Description | Destructor of `IReduction`. It destroys the `IReduction` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `train`<a name="ZH-CN_TOPIC_0000001506495753"></a>

<a name="table7235918388"></a>

| API Definition | ```virtual void train(idx_t n, const float *x) const = 0;``` |
| --- | --- |
| Description | Abstract training interface. This function does not provide a concrete implementation. |
| Input | `idx_t n`: Number of feature vectors in the training set.<br>`const float *x`: Feature-vector data. |
| Output | None |
| Returns | None |
| Constraints | The value of `n` must be in the range 0 < `n` < 1e9. Pointer `x` must be non-null and its length must be `dimIn * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

## `AscendNNInference`<a name="ZH-CN_TOPIC_0000001456375320"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456535204"></a>

Performs inference through a neural network.

### `AscendNNInference`<a name="ZH-CN_TOPIC_0000001456854780"></a>

<a name="table7235918388"></a>

| API Definition | ```AscendNNInference(std::vector<int> deviceList, const char* model, uint64_t modelSize);``` |
| --- | --- |
| Description | Constructor of `AscendNNInference`. It creates `AscendNNInference` and configures the Ascend AI Processor resources on the device side and the model path based on the values in `deviceList`. |
| Input | `std::vector<int> deviceList`: Device IDs on the NPU.<br>`const char* model`: Deep neural network dimensionality reduction model.<br>`uint64_t modelSize`: Size of the deep neural network dimensionality reduction model. |
| Output | None |
| Returns | None |
| Constraints | The valid range of `deviceList` is (0, 32]. <br>`model` must be a valid, effective memory pointer to a deep neural network dimensionality reduction model, and its size must be `modelSize`. The valid range of `modelSize` is (0, 128 MB]. Parameter mismatches may cause model instantiation or inference to fail. Invalid models may harm the system. Ensure that the model source is valid and effective. <br>`dimsIn` ∈ {64, 128, 256, 384, 512, 768, 1024}. <br>`dimsOut` ∈ {32, 64, 96, 128, 256}. <br>`batches` ∈ {1, 2, 4, 8, 16, 32, 64, 128}. |

<a name="table1246213101873"></a>

| API Definition | ```AscendNNInference(const AscendNNInference&) = delete;``` |
| --- | --- |
| Description | Declares the copy constructor of `AscendNNInference` as deleted. Therefore, `AscendNNInference` is a non-copyable type. |
| Input | `const AscendNNInference&`: Constant `AscendNNInference`. |
| Output | None |
| Returns | None |
| Constraints | None |

### `~AscendNNInference`<a name="ZH-CN_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>

| API Definition | ```~AscendNNInference();``` |
| --- | --- |
| Description | Destructor of `AscendNNInference`. It destroys the `AscendNNInference` object and releases resources. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

### `getDimBatch`<a name="ZH-CN_TOPIC_0000001506334797"></a>

<a name="zh-cn_topic_0000001287392566_table7235918388"></a>

| API Definition | ```int getDimBatch() const;``` |
| --- | --- |
| Description | Gets the number of samples or query vectors in a single inference pass. |
| Input | None |
| Output | None |
| Returns | The number of samples or query vectors in a single inference pass. |
| Constraints | None |

### `getInputType`<a name="ZH-CN_TOPIC_0000001456854776"></a>

<a name="zh-cn_topic_0000001340072289_table7235918388"></a>

| API Definition | ```int getInputType() const;``` |
| --- | --- |
| Description | Gets the input data type of the model. |
| Input | None |
| Output | None |
| Returns | Input data type of the model. |
| Constraints | None |

### `getOutputType`<a name="ZH-CN_TOPIC_0000001456854868"></a>

| API Definition | ```int getOutputType() const;``` |
| --- | --- |
| Description | Gets the output data type of the model. |
| Input | None |
| Output | None |
| Returns | Output data type of the model. |
| Constraints | None |

### `getDimIn`<a name="ZH-CN_TOPIC_0000001456535128"></a>

| API Definition | ```int getDimIn() const;``` |
| --- | --- |
| Description | Gets the input data dimension of the model. |
| Input | None |
| Output | None |
| Returns | Input data dimension. |
| Constraints | None |

### `getDimOut`<a name="ZH-CN_TOPIC_0000001456695056"></a>

| API Definition | ```int getDimOut() const;``` |
| --- | --- |
| Description | Gets the output data dimension of the model. |
| Input | None |
| Output | None |
| Returns | Output data dimension of the model. |
| Constraints | None |

### `infer`<a name="ZH-CN_TOPIC_0000001506495709"></a>

| API Definition | ```void infer(size_t n, const char* inputData, char* outputData) const;``` |
| --- | --- |
| Description | Performs inference using the neural network model. |
| Input | `size_t n`: Number of inputs for inference.<br>`const char* inputData`: Feature vectors for inference. |
| Output | `char* outputData`: Feature vector results from inference. |
| Returns | None |
| Constraints | The value of `n` must be in the range 0 < `n` < 1e9. Pointer `inputData` must be non-null and its length must be `dimIn * n`. Pointer `outputData` must be non-null and its length must be `dimOut * n`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. |

### `operator=`<a name="ZH-CN_TOPIC_0000001456535156"></a>

| API Definition | ```AscendNNInference& operator=(const AscendNNInference&) = delete;``` |
| --- | --- |
| Description | Declares the copy assignment operator of `AscendNNInference` as deleted. Therefore, `AscendNNInference` is a non-copyable type. |
| Input | `const AscendNNInference&`: Constant `AscendNNInference`. |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendClonerOptions`<a name="ZH-CN_TOPIC_0000001456854804"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456535196"></a>

Configuration parameters for the `AscendCloner` interface.

**Members<a name="section1372191465013"></a>**

| Member | Type | Description |
|--|--|--|
| reserveVecs | long | Currently unused. Number of features reserved in memory. |
| verbose | bool | Whether to print copy logs. |
| resourceSize | int64_t | Resource pool size. |
| slim | bool | Member variable of `AscendIndexSQConfig`. Whether to dynamically increase memory. The default value is `false`. |
| filterable | bool | Member variable of `AscendIndexSQConfig`. Whether to filter by ID. The default value is `false`. |
| indexMode | uint32_t | Index INT8 retrieval mode. The default value is `0` (`DEFAULT_MODE`). |
| blockSize | uint32_t | `blockSize` configured on the device side. The default value of `BLOCK_SIZE` is `16384 * 16 = 262144`. |

### `AscendClonerOptions`<a name="ZH-CN_TOPIC_0000001506414885"></a>

| API Definition | ```AscendClonerOptions()``` |
| --- | --- |
| Description | Constructor of `AscendClonerOptions`. |
| Input | None |
| Output | None |
| Returns | None |
| Constraints | None |

## `AscendCloner`<a name="ZH-CN_TOPIC_0000001506334577"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456375412"></a>

Index SDK provides an operation that copies retrieval `Index` resources on the NPU to Faiss on the CPU side. The copy process happens in memory. Data loaded in the original NPU `Index` is copied into CPU-side memory, which makes it convenient for users to run retrieval with the same base library on the CPU.

> [!NOTE]
> Some versions of Faiss provide a method for persisting an in-memory `Index` to disk, that is, saving in-memory data to a local drive. When you use Index SDK and Faiss to process sensitive data, pay special attention to the corresponding access control and encryption protection.

### `index_ascend_to_cpu`<a name="ZH-CN_TOPIC_0000001506334821"></a>

| API Definition | ```faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);``` |
| --- | --- |
| Description | Copies retrieval `Index` resources on Ascend and creates a retrieval `Index` on the CPU. |
| Input | `const faiss::Index *ascend_index`: `Index` resource on Ascend. |
| Output | None |
| Returns | A retrieval `Index` on the CPU. |
| Constraints | After you finish using this `index`, remember to `delete` this pointer to release the corresponding memory. |

### `index_cpu_to_ascend`<a name="ZH-CN_TOPIC_0000001456695032"></a>

| API Definition | ```faiss::Index *index_cpu_to_ascend(std::initializer_list<int> devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);``` |
| --- | --- |
| Description | Copies retrieval `Index` resources on the CPU and creates a retrieval `Index` on Ascend. |
| Input | `std::initializer_list<int> devices`: Device IDs to configure on the NPU.<br>`const faiss::Index *index`: Retrieval `Index` resources on the CPU.<br>`const AscendClonerOptions *options = nullptr`: `AscendClonerOptions` resource to configure. |
| Output | None |
| Returns | A retrieval `Index` on Ascend. |
| Constraints | After you finish using this `index`, remember to `delete` this pointer to release the corresponding memory. <br>`devices` must be valid, non-duplicated device IDs, and the maximum number is 64. <br>`index` must be a valid CPU `Index` pointer. |

| API Definition | ```faiss::Index *index_cpu_to_ascend(std::vector<int> devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);``` |
| --- | --- |
| Description | Copies retrieval `Index` resources on the CPU and creates a retrieval `Index` on Ascend. |
| Input | `std::vector<int> devices`: Device IDs to configure on the NPU.<br>`const faiss::Index *index`: Retrieval `Index` resources on the CPU.<br>`const AscendClonerOptions *options = nullptr`: `AscendClonerOptions` resource to configure. |
| Output | None |
| Returns | A retrieval `Index` on Ascend. |
| Constraints | After you finish using this `index`, remember to `delete` this pointer to release the corresponding memory. <br>`devices` must be valid, non-duplicated device IDs, and the maximum number is 64. <br>`index` must be a valid CPU `Index` pointer. |

### `index_int8_ascend_to_cpu`<a name="ZH-CN_TOPIC_0000001506414761"></a>

| API Definition | ```faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);``` |
| --- | --- |
| Description | Copies INT8 retrieval `Index` resources on Ascend and creates a retrieval `Index` on the CPU. |
| Input | `const AscendIndexInt8 *index`: `Index` resource on Ascend. |
| Output | None |
| Returns | A retrieval `Index` on the CPU. |
| Constraints | After you finish using this `index`, remember to `delete` this pointer to release the corresponding memory.<br>`index` must be a valid `AscendIndexInt8` pointer. |

### `index_int8_cpu_to_ascend`<a name="ZH-CN_TOPIC_0000001456375248"></a>

| API Definition | ```AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list<int> devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);``` |
| --- | --- |
| Description | Copies retrieval `Index` resources on the CPU and creates an INT8 retrieval `Index` on Ascend. |
| Input | `std::initializer_list<int> devices`: Device IDs to configure on the NPU.<br>`const faiss::Index *index`: Retrieval `Index` resources on the CPU.<br>`const AscendClonerOptions *options = nullptr`: `AscendClonerOptions` resource to configure. |
| Output | None |
| Returns | An INT8 retrieval `Index` on Ascend. |
| Constraints | After you finish using this `index`, remember to `delete` this pointer to release the corresponding memory. `devices` must be valid, non-duplicated device IDs, and the maximum number is 64. `index` must be a valid CPU `Index` pointer. |

| API Definition | ```AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector<int> devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);``` |
| --- | --- |
| Description | Copies retrieval `Index` resources on the CPU and creates an INT8 retrieval `Index` on Ascend. |
| Input | `std::vector<int> devices`: Device IDs to configure on the NPU.<br>`const faiss::Index *index`: Retrieval `Index` resources on the CPU.<br>`const AscendClonerOptions *options = nullptr`: `AscendClonerOptions` resource to configure. |
| Output | None |
| Returns | An INT8 retrieval `Index` on Ascend. |
| Constraints | After you finish using this `index`, remember to `delete` this pointer to release the corresponding memory. <br>`devices` must be valid, non-duplicated device IDs, and the maximum number is 64. <br>`index` must be a valid CPU `Index` pointer. |

## `DiskPQ`<a name="ZH-CN_TOPIC_0000002382802364"></a>

### Overview<a name="ZH-CN_TOPIC_0000002382647580"></a>

Index SDK provides training and retrieval functions for PQ (Product Quantization) quantization. The PQ interface does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you need to lock before use. Otherwise, the function may behave abnormally.

### `DiskPQParams`<a name="ZH-CN_TOPIC_0000002382807444"></a>

| API Definition | DiskPQParams {<br>int pqChunks = 512;<br>int funcType = 1;<br>int dim = 1;<br>char \*pqTable = nullptr;<br>uint32_t \*offsets = nullptr;<br>char \*tablesTransposed = nullptr;<br>char \*centroids = nullptr;<br>} |
| --- | --- |
| Description | PQ quantization structure. |
| Input | None |
| Output | None |
| Parameter values | `int pqChunks`: Splits the original vector dimension `dim` into `pqChunks` chunks.<br>`int funcType`: Computation standard used for PQ table lookup distance calculation.<br>`int dim`: Original data dimension.<br>`char *pqTable`: Pointer to the codebook data. The default value is `nullptr`.<br>`uint32_t *offsets`: Pointer to the starting and ending dimensions of each chunk in the original dimension. The default value is `nullptr`.<br>`char *tablesTransposed`: Pointer to the transposed form of the codebook data. The default value is `nullptr`.<br>`char *centroids`: Pointer to the mean value of each dimension, used to center the data. The default value is `nullptr`. |
| Parameter constraints | 1 <= `pqChunks` <= `dim`. Smaller `pqChunks` use less memory, but they also reduce accuracy. In general, you are advised to set `pqChunks` to `dim / 8` or `dim / 16`, rounded up in both cases. The default value is 512. <br>The valid range of `funcType` is 1 to 3. 1 indicates L2 distance, 2 indicates IP distance, and 3 indicates cosine distance. The default value is 1. <br>1 <= `dim` <= 2000. The default value is 1. <br>`pqTable` currently supports only the `float` data type, that is, the `Vector` data type in OpenGauss. <br>`tablesTransposed` currently supports only the `float` data type, that is, the `Vector` data type in OpenGauss. |

### `VectorArrayData`<a name="ZH-CN_TOPIC_0000002416326913"></a>

| API Definition | VectorArrayData {<br>int length;<br>int maxlen;<br>int dim;<br>size_t itemsize;<br>char *items;<br>} |
| --- | --- |
| Description | Data encapsulation structure. |
| Input | None |
| Output | None |
| Parameter values | `int length`: Number of vectors stored in the structure.<br>`int maxlen`: Maximum number of vectors stored in the structure.<br>`int dim`: Vector dimension stored in the structure.<br>`size_t itemsize`: Reserved field. Users can choose not to set it.<br>`char *items`: Pointer to the data stored in `VectorArrayData`. The default value is `nullptr`. |
| Parameter constraints | 1 <= `length` <= 100000000. <br>`maxlen` is a reserved field on the OpenGauss side. Non-OpenGauss users can set it to the same value as `length`. <br>1 <= `dim` <= 2000. <br>For different APIs, ensure that `items` points to data of the required size. |

### `ComputePQTable`<a name="ZH-CN_TOPIC_0000002416446741"></a>

| API Definition | ```int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);``` |
| --- | --- |
| Description | Uses the sampled base-library data stored in `sample` to compute the PQ codebook and stores the codebook-related data in the corresponding parameters in `params`. |
| Input | `VectorArrayData *sample`: Pointer to the `VectorArrayData` instance that contains the sampled base-library data. Must not be a null pointer.<br>`DiskPQParams *params`: Pointer to the `DiskPQParams` instance that contains only PQ parameters and no trained PQ data. Must not be a null pointer. |
| Output | None |
| Returns | `int`: `0` indicates that the process is normal. `-1` indicates that the process failed, and the error logs are printed to `cerr`. |
| Constraints | The `sample` data must meet the following requirements:<br>The data pointed to by `items` must be `(8 + dim) * length * sizeof(float)` bytes, which means each vector has 8 bytes of metadata in front of it. When non-OpenGauss users use this API, they need to add 8 bytes of arbitrary data to each vector entry. <br>The `params` members must meet the following requirements: <br>In addition to the range limits described above, `dim` must match the corresponding `dim` field in `sample`. <br>`pqTable` must be `nullptr`. The dynamic library allocates memory with `new[]`, and you must release the allocated memory outside the library with `delete[]`. The allocated memory size is `dim * 256 * sizeof(float)` bytes, where `256` is the number of clusters in each chunk. <br>`offsets` must be `nullptr`. The dynamic library allocates memory with `new[]`, and you must release the allocated memory outside the library with `delete[]`. The allocated memory size is `(pqChunks + 1) * sizeof(uint32_t)` bytes. <br>`tablesTransposed` must be `nullptr`. The dynamic library allocates memory with `new[]`, and you must release the allocated memory outside the library with `delete[]`. The allocated memory size is `dim * 256 * sizeof(float)` bytes. <br>`centroids` must be `nullptr`. The dynamic library allocates memory with `new[]`, and you must release the allocated memory outside the library with `delete[]`. The allocated memory size is `dim * sizeof(float)` bytes. |

### `ComputeVectorPQCode`<a name="ZH-CN_TOPIC_0000002382647584"></a>

| API Definition | ```int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);``` |
| --- | --- |
| Description | Uses the `params` filled with PQ data to quantize the base-library data in `baseData` and writes the quantized data into the buffer pointed to by `pqCode`. |
| Input | `VectorArrayData *baseData`: Pointer to the `VectorArrayData` instance that contains the base-library data. Must not be a null pointer. You can determine the size of the base-library data in `baseData` externally based on your memory limits.<br>`const DiskPQParams *params`: Pointer to the `DiskPQParams` instance that contains PQ parameters and trained PQ data. Must not be a null pointer. |
| Output | `uint8_t *pqCode`: Pointer that receives the compressed base-library vectors. Must not be a null pointer. |
| Returns | `int`: `0` indicates that the process is normal. `-1` indicates that the process failed, and the error logs are printed to `cerr`. |
| Constraints | The `baseData` data must meet the following requirements:<br>The data pointed to by `items` must be `length * dim * sizeof(float)` bytes. Note that, unlike the `ComputePQTable` interface, you do not need to add placeholder metadata before each data entry. <br>The `params` members must meet the following requirements: <br>In addition to the range limits described above, `dim` must match the corresponding `dim` field in `baseData`. <br>`pqTable` must point to codebook data whose size is `dim * 256 * sizeof(float)` bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <br>`offsets` must point to `offsets` data whose size is `(pqChunks + 1) * sizeof(uint32_t)` bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <br>There is no requirement for `tablesTransposed`. <br>`centroids` must point to `centroids` data whose size is `dim * sizeof(float)` bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <br>Ensure that the space pointed to by `pqCode` is at least `length * pqChunks` bytes. Here, `length` is the `VectorArrayData` parameter and `pqChunks` is the `DiskPQParams` parameter. |

### `GetPQDistanceTable`<a name="ZH-CN_TOPIC_0000002382807448"></a>

| API Definition | ```int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);``` |
| --- | --- |
| Description | Uses the `params` filled with PQ data to perform ADC PQ distance calculation on the query data pointed to by `vec` and writes the PQ distance table into the buffer pointed to by `pqDistanceTable`. |
| Input | `char *vec`: Pointer to the query data to calculate.<br>`const DiskPQParams *params`: Pointer to the `DiskPQParams` instance that contains PQ parameters and trained PQ data. Must not be a null pointer. |
| Output | `float *pqDistanceTable`: Pointer that receives the distances between the query and each centroid in each chunk. |
| Returns | `int`: `0` indicates that the process is normal. `-1` indicates that the process failed, and the error logs are printed to `cerr`. |
| Constraints | Ensure that the space pointed to by `vec` is at least `dim * sizeof(float)` bytes. Currently, only the `float` data type is supported, that is, the `Vector` data type in OpenGauss. <br>The `params` members must meet the following requirements: <br>There is no requirement for the value pointed to by `pqTable`. <br>`offsets` must point to `offsets` data whose size is `(pqChunks + 1) * sizeof(uint32_t)` bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <br>`tablesTransposed` must point to codebook data whose size is `dim * 256 * sizeof(float)` bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <br>`centroids` must point to `centroids` data whose size is `dim * sizeof(float)` bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <br>Ensure that the space pointed to by `pqDistanceTable` is at least `pqChunks * 256 * sizeof(float)` bytes. |

### `GetPQDistance`<a name="ZH-CN_TOPIC_0000002416326917"></a>

| API Definition | ```int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &pqDistance);``` |
| --- | --- |
| Description | Uses the compressed code data corresponding to the base-library vector pointed to by `basecode` and the `pqDistanceTable` obtained from the `GetPQDistanceTable` API to calculate the PQ distance between the query and that base-library vector. |
| Input | `const uint8_t *basecode`: Pointer to the compressed code data corresponding to a base-library vector.<br>`const DiskPQParams *params`: Pointer to the `DiskPQParams` instance with the `pqChunks` value filled in. Must not be a null pointer.<br>`const float *pqDistanceTable`: Pointer to the ADC PQ distance table corresponding to the query. |
| Output | `float &pqDistance`: Reference to the final output PQ distance value. |
| Returns | `int`: `0` indicates that the process is normal. `-1` indicates that the process failed, and the error logs are printed to `cerr`. |
| Constraints | Ensure that the data pointed to by `basecode` is at least `pqChunks` bytes. <br>In `params`, you only need to fill in the `pqChunks` value, and it must match the `pqChunks` value mentioned for `basecode`. <br>Ensure that the data pointed to by `pqDistanceTable` is at least `pqChunks * 256 * sizeof(float)` bytes. <br>The interface does not zero `pqDistance` before use. The final result of `pqDistance` is the original `pqDistance` value plus the PQ distance between the query and `basecode`. Therefore, an input value of `0` is recommended. |

## `GetVersionInfo`<a name="ZH-CN_TOPIC_0000001456535080"></a>

| API Definition | ```std::string GetVersionInfo();``` |
| --- | --- |
| Description | Gets version information. It retrieves the corresponding version information based on the `MX_INDEX_HOME` environment variable. This environment variable is set automatically when the software package is installed, so no modification is needed. |
| Input | None |
| Output | None |
| Returns | Version information. |
| Constraints | None |
