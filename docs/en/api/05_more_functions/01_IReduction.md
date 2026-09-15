# IReduction<a name="en-us_TOPIC_0000001456694992"></a>

## Function Description<a name="en-us_TOPIC_0000001506615161"></a>

`IReduction` is the unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports the `PCAR` and `NN` dimensionality reduction algorithms.

## `CreateReduction`<a name="en-us_TOPIC_0000001456695108"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>IReduction *CreateReduction(std::string typeName, const ReductionConfig &amp;config);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Creates a specific dimensionality reduction algorithm.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::string typeName</code>: Dimensionality reduction algorithm parameter. Valid values are <code>{&quot;NN&quot;, &quot;PCAR&quot;}</code>.<br><code>ReductionConfig &amp;config</code>: Dimensionality reduction configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle"><code>IReduction *CreateReduction</code>: Created dimensionality reduction instance.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Currently, only the <code>NN</code> and <code>PCAR</code> dimensionality reduction parameters are supported. Using any other parameter causes an exception.<br>After you finish using this instance, remember to <code>delete</code> this pointer to release the corresponding memory.</td></tr>
</tbody></table>

## `reduce`<a name="en-us_TOPIC_0000001456375280"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void reduce(idx_t n, const float *x, float *res) const = 0;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Dimensionality reduction interface. This function does not provide a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of inputs for inference.<br><code>const float *x</code>: Feature vectors for inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle"><code>float *res</code>: Feature-vector results from inference.</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9. Pointer <code>x</code> must be non-null and its length must be <code>dimIn * n</code>. Pointer <code>res</code> must be non-null and its length must be <code>dimOut * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

## `ReductionConfig`<a name="en-us_TOPIC_0000001456375264"></a>

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

## `~IReduction`<a name="en-us_TOPIC_0000001714244661"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~IReduction() = default;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>IReduction</code>. It destroys the <code>IReduction</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000001506495753"></a>

<a name="table7235918388"></a>
<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual void train(idx_t n, const float *x) const = 0;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Abstract training interface. This function does not provide a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n</code>: Number of feature vectors in the training set.<br><code>const float *x</code>: Feature-vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9. Pointer <code>x</code> must be non-null and its length must be <code>dimIn * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>
