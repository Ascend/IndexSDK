# `AscendNNInference`<a name="en-us_TOPIC_0000001456375320"></a>

## Function Description<a name="en-us_TOPIC_0000001456535204"></a>

Performs inference through a neural network.

## `AscendNNInference`<a name="en-us_TOPIC_0000001456854780"></a>

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

## `~AscendNNInference`<a name="en-us_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>~AscendNNInference();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendNNInference</code>. It destroys the <code>AscendNNInference</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getDimBatch`<a name="en-us_TOPIC_0000001506334797"></a>

<a name="en-us_topic_0000001287392566_table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDimBatch() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the number of samples or query vectors in a single inference pass.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">The number of samples or query vectors in a single inference pass.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getInputType`<a name="en-us_TOPIC_0000001456854776"></a>

<a name="en-us_topic_0000001340072289_table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getInputType() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the input data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Input data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getOutputType`<a name="en-us_TOPIC_0000001456854868"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getOutputType() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the output data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Output data type of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getDimIn`<a name="en-us_TOPIC_0000001456535128"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDimIn() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the input data dimension of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Input data dimension.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `getDimOut`<a name="en-us_TOPIC_0000001456695056"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>int getDimOut() const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Gets the output data dimension of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">Output data dimension of the model.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `infer`<a name="en-us_TOPIC_0000001506495709"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void infer(size_t n, const char* inputData, char* outputData) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Performs inference using the neural network model.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>size_t n</code>: Number of inputs for inference.<br><code>const char* inputData</code>: Feature vectors for inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>char* outputData</code>: Feature vector results from inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9. Pointer <code>inputData</code> must be non-null and its length must be <code>dimIn * n</code>. Pointer <code>outputData</code> must be non-null and its length must be <code>dimOut * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001456535156"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendNNInference&amp; operator=(const AscendNNInference&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declares the copy assignment operator of <code>AscendNNInference</code> as deleted. Therefore, <code>AscendNNInference</code> is a non-copyable type.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendNNInference&amp;</code>: Constant <code>AscendNNInference</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>
