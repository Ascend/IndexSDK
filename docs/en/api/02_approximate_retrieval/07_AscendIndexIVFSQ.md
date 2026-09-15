# `AscendIndexIVFSQ`<a name="en-us_TOPIC_0000001506334625"></a>

## Overview<a name="en-us_TOPIC_0000001456694964"></a>

The `AscendIndexIVFSQ` class uses IVF for acceleration and is a two-stage approximate retrieval algorithm.

It supports concurrent multithreaded calls. You need to set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, `export MX_INDEX_MULTITHREAD=1`. Setting it to any other value or leaving it unset means that multithreading is disabled. Current feature retrieval internally uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep growing, so you are advised to use fixed threads to run retrieval tasks.

## `AscendIndexIVFSQ`<a name="en-us_TOPIC_0000001506414893"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQ</code>. It creates a retrieval <code>Index</code> on Ascend based on an existing <code>index</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code>.<br><code>AscendIndexIVFSQConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

<a name="table1823217151014"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(int dims, int nlist, faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true, AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQ</code>. It creates an <code>AscendIndexIVFSQ</code>, and the device-side resources are set according to the values configured in <code>config</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims:</code> Dimension of the feature vectors managed by <code>AscendIndexIVFSQ</code>.<br><code>int nlist:</code> Number of cluster centroids. This parameter corresponds to <code>coarse_centroid_num</code> in the operator generation script.<br><code>faiss::ScalarQuantizer::QuantizerType qtype:</code> Quantizer type of <code>AscendIndexIVFSQ</code>.<br><code>faiss::MetricType metric:</code> Distance metric used by <code>AscendIndex</code> for feature vector similarity search.<br><code>bool encodeResidual:</code> Whether to encode residuals.<br><code>AscendIndexIVFSQConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {64, 128, 256, 384, 512}. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>qtype = ScalarQuantizer::QuantizerType::QT_8bit</code>, and only <code>ScalarQuantizer::QuantizerType::QT_8bit</code> is supported. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.<br>Note:<br>Currently, when <code>metric = faiss::MetricType::METRIC_INNER_PRODUCT</code>, <code>encodeResidual</code> only supports <code>false</code>. That is, the IVFSQ method with residual encoding is not currently supported. When <code>encodeResidual</code> is <code>true</code>, the code can run successfully, but there is an accuracy issue.</td></tr>
</tbody></table>

<a name="table134501935171012"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSQ</code>. It creates an <code>AscendIndexIVFSQ</code>, and the device-side resources are set according to the values configured in <code>config</code>. This API does not perform initialization. The subclass performs the initialization-related work. This API will be deprecated later, so do not use it.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>int dims:</code> Dimension of the feature vectors managed by <code>AscendIndexIVFSQ</code>.<br><code>int nlist:</code> Number of cluster centroids. This parameter corresponds to <code>coarse_centroid_num</code> in the operator generation script.<br><code>faiss::MetricType metric:</code> Distance metric used by <code>AscendIndex</code> for feature vector similarity search.<br><code>AscendIndexIVFSQConfig config:</code> Device-side resource configuration.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>dims</code> ∈ {64, 128, 256, 384, 512}. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>metric</code> ∈ {<code>faiss::MetricType::METRIC_L2</code>, <code>faiss::MetricType::METRIC_INNER_PRODUCT</code>}.</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ(const AscendIndexIVFSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> copy constructor as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQ&amp;:</code> Constant <code>AscendIndexIVFSQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `~AscendIndexIVFSQ`<a name="en-us_TOPIC_0000001456534936"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexIVFSQ();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Destructor for <code>AscendIndexIVFSQ</code>. It destroys the <code>AscendIndexIVFSQ</code> object and releases resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `copyFrom`<a name="en-us_TOPIC_0000001456375244"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIVFScalarQuantizer *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy an existing <code>index</code> to Ascend based on <code>AscendIndexIVFSQ</code>, while keeping the original device-side resource configuration of <code>AscendIndex</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer.<br><code>index-&gt;d</code> ∈ {256}. <code>index-&gt;sq.d</code> ∈ {32, 64, 128}. The dimension of <code>index</code> must be greater than the dimension of <code>index-&gt;sq</code>, and it must be divisible by the dimension of <code>index-&gt;sq</code>. Do not call this API on an updated object.</td></tr>
</tbody></table>

## `copyTo`<a name="en-us_TOPIC_0000001506334649"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIVFScalarQuantizer *index) const;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copy the retrieval resources of <code>AscendIndexIVFSQ</code> to the CPU side.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>faiss::IndexIVFScalarQuantizer *index:</code> CPU-side <code>Index</code> resources.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>index</code> must be a valid CPU <code>Index</code> pointer. The user is responsible for freeing the memory occupied by the <code>Index</code>.</td></tr>
</tbody></table>

## `operator=`<a name="en-us_TOPIC_0000001456854860"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexIVFSQ&amp; operator=(const AscendIndexIVFSQ&amp;) = delete;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Declare this <code>Index</code> assignment operator as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexIVFSQ&amp;:</code> Constant <code>AscendIndexIVFSQ</code>.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

## `train`<a name="en-us_TOPIC_0000001456854976"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Train <code>AscendIndexIVFSQ</code>. This class inherits the relevant APIs in <code>AscendIndex</code> and provides a concrete implementation.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>idx_t n:</code> Number of feature vectors in the training set.<br><code>const float *x:</code> Feature vector data.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">Training uses k-means clustering, and a small training set may affect query accuracy. The value range of <code>n</code> here is <code>0 &lt; n &lt; 1e9</code>. The pointer <code>x</code> must be a non-null pointer, and its length should be <code>dims * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>
