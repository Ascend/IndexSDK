# Other Functions<a name="ZH-CN_TOPIC_0000001482684458"></a>

## `IReduction`<a name="ZH-CN_TOPIC_0000001456694992"></a>

### Overview<a name="ZH-CN_TOPIC_0000001506615161"></a>

`IReduction` is the unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports the `PCAR` and `NN` dimensionality reduction algorithms.

### `CreateReduction`<a name="ZH-CN_TOPIC_0000001456695108"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IReduction *CreateReduction(std::string typeName, const ReductionConfig &amp;config);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Creates a specific dimensionality reduction algorithm.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::string typeName</code>: Dimensionality reduction algorithm parameter. Valid values are <code>{&quot;NN&quot;, &quot;PCAR&quot;}</code>.<br><code>ReductionConfig &amp;config</code>: Dimensionality reduction configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>IReduction *CreateReduction</code>: Created dimensionality reduction instance.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Currently, only the <code>NN</code> and <code>PCAR</code> dimensionality reduction parameters are supported. Using any other parameter causes an exception.<br>After you finish using this instance, remember to <code>delete</code> this pointer to release the corresponding memory.</td></tr>
</tbody></table>

### `reduce`<a name="ZH-CN_TOPIC_0000001456375280"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void reduce(idx_t n, const float *x, float *res) const = 0;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Dimensionality reduction interface. This function does not provide a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of inputs for inference.<br><code>const float *x</code>: Feature vectors for inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *res</code>: Feature-vector results from inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9. Pointer <code>x</code> must be non-null and its length must be <code>dimIn * n</code>. Pointer <code>res</code> must be non-null and its length must be <code>dimOut * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `ReductionConfig`<a name="ZH-CN_TOPIC_0000001456375264"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">dimIn</td><td valign="middle">int</td><td valign="middle">Input feature dimension, that is, the dimension before reduction. <code>PCAR</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">dimOut</td><td valign="middle">int</td><td valign="middle">Output feature dimension, that is, the dimension after reduction. <code>PCAR</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">eigenPower</td><td valign="middle">float</td><td valign="middle">Power of the singular values. <code>PCAR</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">randomRotation</td><td valign="middle">bool</td><td valign="middle">Whether to perform random rotation. <code>PCAR</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device-side resource configuration. <code>NN</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">model</td><td valign="middle">const char *</td><td valign="middle">Neural network dimensionality reduction model. <code>NN</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">modelSize</td><td valign="middle">uint64_t</td><td valign="middle">Model size. <code>NN</code> requires this parameter.</td></tr>
</tbody></table>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline ReductionConfig(int dimIn, int dimOut, float eigenPower, bool randomRotation);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>ReductionConfig</code>. Use this function when you use <code>PCAR</code> dimensionality reduction.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dimIn</code>: Input feature dimension, that is, the dimension before reduction. <code>PCAR</code> requires this parameter.<br><code>int dimOut</code>: Output feature dimension, that is, the dimension after reduction. <code>PCAR</code> requires this parameter.<br><code>float eigenPower</code>: Power of the singular values. <code>PCAR</code> requires this parameter.<br><code>bool randomRotation</code>: Whether to perform random rotation. <code>PCAR</code> requires this parameter.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">When you use different dimensionality reduction algorithms, configure the corresponding parameters, and ensure that the dimension after reduction satisfies the dimension limit of the downstream index that uses the reduced data. When you use <code>PCAR</code> dimensionality reduction, ensure that <code>dimOut</code> &gt; 0 and <code>dimIn</code> &gt;= <code>dimOut</code>. The range of <code>eigenPower</code> is [-0.5, 0].</td></tr>
</tbody></table>

<a name="table2034112619"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline ReductionConfig(std::vector&lt;int&gt; deviceList, const char *model, uint64_t modelSize);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor of <code>ReductionConfig</code>. Use this function when you use <code>NN</code> dimensionality reduction.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; deviceList</code>: Device-side resource configuration.<br><code>const char *model</code>: Neural network dimensionality reduction model.<br><code>uint64_t modelSize</code>: Model size.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>deviceList</code> is (0, 32]. When you use different dimensionality reduction algorithms, configure the corresponding parameters, and ensure that the dimension after reduction satisfies the dimension limit of the downstream index that uses the reduced data. <code>model</code> must be a valid, effective memory pointer to a deep neural network dimensionality reduction model, and its size must be <code>modelSize</code>. The valid range of <code>modelSize</code> is (0, 128 MB]. Parameter mismatches may cause model instantiation or inference to fail. Invalid models may harm the system. Ensure that the model source is valid and effective. <code>dimsIn</code> ∈ {64, 128, 256, 384, 512, 768, 1024}. <code>dimsOut</code> ∈ {32, 64, 96, 128, 256}. <code>batches</code> ∈ {1, 2, 4, 8, 16, 32, 64, 128}.</td></tr>
</tbody></table>

### `~IReduction`<a name="ZH-CN_TOPIC_0000001714244661"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~IReduction() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>IReduction</code>. It destroys the <code>IReduction</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `train`<a name="ZH-CN_TOPIC_0000001506495753"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void train(idx_t n, const float *x) const = 0;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Abstract training interface. This function does not provide a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9. Pointer <code>x</code> must be non-null and its length must be <code>dimIn * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `AscendNNInference`<a name="ZH-CN_TOPIC_0000001456375320"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456535204"></a>

Performs inference through a neural network.

### `AscendNNInference`<a name="ZH-CN_TOPIC_0000001456854780"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendNNInference(std::vector&lt;int&gt; deviceList, const char* model, uint64_t modelSize);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendNNInference</code>. It creates <code>AscendNNInference</code> and configures the Ascend AI Processor resources on the device side and the model path based on the values in <code>deviceList</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; deviceList</code>: Device IDs on the NPU.<br><code>const char* model</code>: Deep neural network dimensionality reduction model.<br><code>uint64_t modelSize</code>: Size of the deep neural network dimensionality reduction model.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The valid range of <code>deviceList</code> is (0, 32]. <code>model</code> must be a valid, effective memory pointer to a deep neural network dimensionality reduction model, and its size must be <code>modelSize</code>. The valid range of <code>modelSize</code> is (0, 128 MB]. Parameter mismatches may cause model instantiation or inference to fail. Invalid models may harm the system. Ensure that the model source is valid and effective. <code>dimsIn</code> ∈ {64, 128, 256, 384, 512, 768, 1024}. <code>dimsOut</code> ∈ {32, 64, 96, 128, 256}. <code>batches</code> ∈ {1, 2, 4, 8, 16, 32, 64, 128}.</td></tr>
</tbody></table>

<a name="table1246213101873"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendNNInference(const AscendNNInference&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy constructor of <code>AscendNNInference</code> as deleted. Therefore, <code>AscendNNInference</code> is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendNNInference&amp;</code>: Constant <code>AscendNNInference</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendNNInference`<a name="ZH-CN_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendNNInference();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendNNInference</code>. It destroys the <code>AscendNNInference</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getDimBatch`<a name="ZH-CN_TOPIC_0000001506334797"></a>

<a name="zh-cn_topic_0000001287392566_table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDimBatch() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the number of samples or query vectors in a single inference pass.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">The number of samples or query vectors in a single inference pass.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getInputType`<a name="ZH-CN_TOPIC_0000001456854776"></a>

<a name="zh-cn_topic_0000001340072289_table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getInputType() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the input data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Input data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getOutputType`<a name="ZH-CN_TOPIC_0000001456854868"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getOutputType() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the output data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Output data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getDimIn`<a name="ZH-CN_TOPIC_0000001456535128"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDimIn() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the input data dimension of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Input data dimension.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `getDimOut`<a name="ZH-CN_TOPIC_0000001456695056"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDimOut() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the output data dimension of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Output data dimension of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `infer`<a name="ZH-CN_TOPIC_0000001506495709"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void infer(size_t n, const char* inputData, char* outputData) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs inference using the neural network model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n</code>: Number of inputs for inference.<br><code>const char* inputData</code>: Feature vectors for inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>char* outputData</code>: Feature vector results from inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9. Pointer <code>inputData</code> must be non-null and its length must be <code>dimIn * n</code>. Pointer <code>outputData</code> must be non-null and its length must be <code>dimOut * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `operator=`<a name="ZH-CN_TOPIC_0000001456535156"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendNNInference&amp; operator=(const AscendNNInference&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of <code>AscendNNInference</code> as deleted. Therefore, <code>AscendNNInference</code> is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendNNInference&amp;</code>: Constant <code>AscendNNInference</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendClonerOptions`<a name="ZH-CN_TOPIC_0000001456854804"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456535196"></a>

Configuration parameters for the `AscendCloner` interface.

**Members<a name="section1372191465013"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">Member</td><td valign="middle">Type</td><td valign="middle">Description</td></tr>
<tr><td width="210" align="center" valign="middle">reserveVecs</td><td valign="middle">long</td><td valign="middle">Currently unused. Number of features reserved in memory.</td></tr>
<tr><td width="210" align="center" valign="middle">verbose</td><td valign="middle">bool</td><td valign="middle">Whether to print copy logs.</td></tr>
<tr><td width="210" align="center" valign="middle">resourceSize</td><td valign="middle">int64_t</td><td valign="middle">Resource pool size.</td></tr>
<tr><td width="210" align="center" valign="middle">slim</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexSQConfig</code>. Whether to dynamically increase memory. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">filterable</td><td valign="middle">bool</td><td valign="middle">Member variable of <code>AscendIndexSQConfig</code>. Whether to filter by ID. The default value is <code>false</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">indexMode</td><td valign="middle">uint32_t</td><td valign="middle">Index INT8 retrieval mode. The default value is <code>0</code> (<code>DEFAULT_MODE</code>).</td></tr>
<tr><td width="210" align="center" valign="middle">blockSize</td><td valign="middle">uint32_t</td><td valign="middle"><code>blockSize</code> configured on the device side. The default value of <code>BLOCK_SIZE</code> is <code>16384 * 16 = 262144</code>.</td></tr>
</tbody></table>

### `AscendClonerOptions`<a name="ZH-CN_TOPIC_0000001506414885"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendClonerOptions()</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendClonerOptions</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `AscendCloner`<a name="ZH-CN_TOPIC_0000001506334577"></a>

### Overview<a name="ZH-CN_TOPIC_0000001456375412"></a>

Index SDK provides an operation that copies retrieval `Index` resources on the NPU to Faiss on the CPU side. The copy process happens in memory. Data loaded in the original NPU `Index` is copied into CPU-side memory, which makes it convenient for users to run retrieval with the same base library on the CPU.

> [!NOTE]
> Some versions of Faiss provide a method for persisting an in-memory `Index` to disk, that is, saving in-memory data to a local drive. When you use Index SDK and Faiss to process sensitive data, pay special attention to the corresponding access control and encryption protection.

### `index_ascend_to_cpu`<a name="ZH-CN_TOPIC_0000001506334821"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on Ascend and creates a retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::Index *ascend_index</code>: <code>Index</code> resource on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory.</td></tr>
</tbody></table>

### `index_cpu_to_ascend`<a name="ZH-CN_TOPIC_0000001456695032"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

### `index_int8_ascend_to_cpu`<a name="ZH-CN_TOPIC_0000001506414761"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies INT8 retrieval <code>Index</code> resources on Ascend and creates a retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8 *index</code>: <code>Index</code> resource on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>index</code> must be a valid <code>AscendIndexInt8</code> pointer.</td></tr>
</tbody></table>

### `index_int8_cpu_to_ascend`<a name="ZH-CN_TOPIC_0000001456375248"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates an INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">An INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates an INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">An INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

## `DiskPQ`<a name="ZH-CN_TOPIC_0000002382802364"></a>

### Overview<a name="ZH-CN_TOPIC_0000002382647580"></a>

Index SDK provides training and retrieval functions for PQ (Product Quantization) quantization. The PQ interface does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you need to lock before use. Otherwise, the function may behave abnormally.

### `DiskPQParams`<a name="ZH-CN_TOPIC_0000002382807444"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle">DiskPQParams {<br>int pqChunks = 512;<br>int funcType = 1;<br>int dim = 1;<br>char \*pqTable = nullptr;<br>uint32_t \*offsets = nullptr;<br>char \*tablesTransposed = nullptr;<br>char \*centroids = nullptr;<br>}</td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">PQ quantization structure.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter values</td><td valign="middle"><code>int pqChunks</code>: Splits the original vector dimension <code>dim</code> into <code>pqChunks</code> chunks.<br><code>int funcType</code>: Computation standard used for PQ table lookup distance calculation.<br><code>int dim</code>: Original data dimension.<br><code>char *pqTable</code>: Pointer to the codebook data. The default value is <code>nullptr</code>.<br><code>uint32_t *offsets</code>: Pointer to the starting and ending dimensions of each chunk in the original dimension. The default value is <code>nullptr</code>.<br><code>char *tablesTransposed</code>: Pointer to the transposed form of the codebook data. The default value is <code>nullptr</code>.<br><code>char *centroids</code>: Pointer to the mean value of each dimension, used to center the data. The default value is <code>nullptr</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter constraints</td><td valign="middle">1 &lt;= <code>pqChunks</code> &lt;= <code>dim</code>. Smaller <code>pqChunks</code> use less memory, but they also reduce accuracy. In general, you are advised to set <code>pqChunks</code> to <code>dim / 8</code> or <code>dim / 16</code>, rounded up in both cases. The default value is 512. The valid range of <code>funcType</code> is 1 to 3. 1 indicates L2 distance, 2 indicates IP distance, and 3 indicates cosine distance. The default value is 1. 1 &lt;= <code>dim</code> &lt;= 2000. The default value is 1. <code>pqTable</code> currently supports only the <code>float</code> data type, that is, the <code>Vector</code> data type in OpenGauss. <code>tablesTransposed</code> currently supports only the <code>float</code> data type, that is, the <code>Vector</code> data type in OpenGauss.</td></tr>
</tbody></table>

### `VectorArrayData`<a name="ZH-CN_TOPIC_0000002416326913"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle">VectorArrayData {<br>int length;<br>int maxlen;<br>int dim;<br>size_t itemsize;<br>char *items;<br>}</td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Data encapsulation structure.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter values</td><td valign="middle"><code>int length</code>: Number of vectors stored in the structure.<br><code>int maxlen</code>: Maximum number of vectors stored in the structure.<br><code>int dim</code>: Vector dimension stored in the structure.<br><code>size_t itemsize</code>: Reserved field. Users can choose not to set it.<br><code>char *items</code>: Pointer to the data stored in <code>VectorArrayData</code>. The default value is <code>nullptr</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Parameter constraints</td><td valign="middle">1 &lt;= <code>length</code> &lt;= 100000000. <code>maxlen</code> is a reserved field on the OpenGauss side. Non-OpenGauss users can set it to the same value as <code>length</code>. 1 &lt;= <code>dim</code> &lt;= 2000. For different APIs, ensure that <code>items</code> points to data of the required size.</td></tr>
</tbody></table>

### `ComputePQTable`<a name="ZH-CN_TOPIC_0000002416446741"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Uses the sampled base-library data stored in <code>sample</code> to compute the PQ codebook and stores the codebook-related data in the corresponding parameters in <code>params</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>VectorArrayData *sample</code>: Pointer to the <code>VectorArrayData</code> instance that contains the sampled base-library data. Must not be a null pointer.<br><code>DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance that contains only PQ parameters and no trained PQ data. Must not be a null pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The <code>sample</code> data must meet the following requirements:<br>The data pointed to by <code>items</code> must be <code>(8 + dim) * length * sizeof(float)</code> bytes, which means each vector has 8 bytes of metadata in front of it. When non-OpenGauss users use this API, they need to add 8 bytes of arbitrary data to each vector entry. The <code>params</code> members must meet the following requirements: In addition to the range limits described above, <code>dim</code> must match the corresponding <code>dim</code> field in <code>sample</code>. <code>pqTable</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new[]</code>, and you must release the allocated memory outside the library with <code>delete[]</code>. The allocated memory size is <code>dim * 256 * sizeof(float)</code> bytes, where <code>256</code> is the number of clusters in each chunk. <code>offsets</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new[]</code>, and you must release the allocated memory outside the library with <code>delete[]</code>. The allocated memory size is <code>(pqChunks + 1) * sizeof(uint32_t)</code> bytes. <code>tablesTransposed</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new[]</code>, and you must release the allocated memory outside the library with <code>delete[]</code>. The allocated memory size is <code>dim * 256 * sizeof(float)</code> bytes. <code>centroids</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new[]</code>, and you must release the allocated memory outside the library with <code>delete[]</code>. The allocated memory size is <code>dim * sizeof(float)</code> bytes.</td></tr>
</tbody></table>

### `ComputeVectorPQCode`<a name="ZH-CN_TOPIC_0000002382647584"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Uses the <code>params</code> filled with PQ data to quantize the base-library data in <code>baseData</code> and writes the quantized data into the buffer pointed to by <code>pqCode</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>VectorArrayData *baseData</code>: Pointer to the <code>VectorArrayData</code> instance that contains the base-library data. Must not be a null pointer. You can determine the size of the base-library data in <code>baseData</code> externally based on your memory limits.<br><code>const DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance that contains PQ parameters and trained PQ data. Must not be a null pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>uint8_t *pqCode</code>: Pointer that receives the compressed base-library vectors. Must not be a null pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The <code>baseData</code> data must meet the following requirements:<br>The data pointed to by <code>items</code> must be <code>length * dim * sizeof(float)</code> bytes. Note that, unlike the <code>ComputePQTable</code> interface, you do not need to add placeholder metadata before each data entry. The <code>params</code> members must meet the following requirements: In addition to the range limits described above, <code>dim</code> must match the corresponding <code>dim</code> field in <code>baseData</code>. <code>pqTable</code> must point to codebook data whose size is <code>dim * 256 * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <code>offsets</code> must point to <code>offsets</code> data whose size is <code>(pqChunks + 1) * sizeof(uint32_t)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. There is no requirement for <code>tablesTransposed</code>. <code>centroids</code> must point to <code>centroids</code> data whose size is <code>dim * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. Ensure that the space pointed to by <code>pqCode</code> is at least <code>length * pqChunks</code> bytes. Here, <code>length</code> is the <code>VectorArrayData</code> parameter and <code>pqChunks</code> is the <code>DiskPQParams</code> parameter.</td></tr>
</tbody></table>

### `GetPQDistanceTable`<a name="ZH-CN_TOPIC_0000002382807448"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Uses the <code>params</code> filled with PQ data to perform ADC PQ distance calculation on the query data pointed to by <code>vec</code> and writes the PQ distance table into the buffer pointed to by <code>pqDistanceTable</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>char *vec</code>: Pointer to the query data to calculate.<br><code>const DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance that contains PQ parameters and trained PQ data. Must not be a null pointer.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *pqDistanceTable</code>: Pointer that receives the distances between the query and each centroid in each chunk.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the space pointed to by <code>vec</code> is at least <code>dim * sizeof(float)</code> bytes. Currently, only the <code>float</code> data type is supported, that is, the <code>Vector</code> data type in OpenGauss. The <code>params</code> members must meet the following requirements: There is no requirement for the value pointed to by <code>pqTable</code>. <code>offsets</code> must point to <code>offsets</code> data whose size is <code>(pqChunks + 1) * sizeof(uint32_t)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <code>tablesTransposed</code> must point to codebook data whose size is <code>dim * 256 * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. <code>centroids</code> must point to <code>centroids</code> data whose size is <code>dim * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur. Ensure that the space pointed to by <code>pqDistanceTable</code> is at least <code>pqChunks * 256 * sizeof(float)</code> bytes.</td></tr>
</tbody></table>

### `GetPQDistance`<a name="ZH-CN_TOPIC_0000002416326917"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &amp;pqDistance);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Uses the compressed code data corresponding to the base-library vector pointed to by <code>basecode</code> and the <code>pqDistanceTable</code> obtained from the <code>GetPQDistanceTable</code> API to calculate the PQ distance between the query and that base-library vector.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const uint8_t *basecode</code>: Pointer to the compressed code data corresponding to a base-library vector.<br><code>const DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance with the <code>pqChunks</code> value filled in. Must not be a null pointer.<br><code>const float *pqDistanceTable</code>: Pointer to the ADC PQ distance table corresponding to the query.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float &amp;pqDistance</code>: Reference to the final output PQ distance value.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Ensure that the data pointed to by <code>basecode</code> is at least <code>pqChunks</code> bytes. In <code>params</code>, you only need to fill in the <code>pqChunks</code> value, and it must match the <code>pqChunks</code> value mentioned for <code>basecode</code>. Ensure that the data pointed to by <code>pqDistanceTable</code> is at least <code>pqChunks * 256 * sizeof(float)</code> bytes. The interface does not zero <code>pqDistance</code> before use. The final result of <code>pqDistance</code> is the original <code>pqDistance</code> value plus the PQ distance between the query and <code>basecode</code>. Therefore, an input value of <code>0</code> is recommended.</td></tr>
</tbody></table>

## `GetVersionInfo`<a name="ZH-CN_TOPIC_0000001456535080"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>std::string GetVersionInfo();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets version information. It retrieves the corresponding version information based on the <code>MX_INDEX_HOME</code> environment variable. This environment variable is set automatically when the software package is installed, so no modification is needed.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Version information.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
