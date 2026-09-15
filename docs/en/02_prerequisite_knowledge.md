# Background Knowledge<a name="en-us_TOPIC_0000001985832240"></a>

## Glossary<a name="en-us_TOPIC_0000001985832241"></a>

<a name="table_glossary"></a>
<table><tbody>
<tr><th width="20%">Term</th><th width="80%">Description</th></tr>
<tr><td><b>Flat</b></td><td>Brute-force search. No complex index structure is built. The query vector is compared with every vector in the base library to calculate distances. It provides 100% recall (exact results) but has high computational overhead and latency. It is commonly used for small-scale datasets or as an accuracy baseline for other algorithms.</td></tr>
<tr><td><b>INT8</b></td><td>8-bit integer format. A low-precision data type that reduces memory usage by 75% compared with standard FP32 (32-bit floating-point) and improves computational throughput. It is commonly used as a quantized data storage format to balance performance and accuracy in hardware-constrained scenarios.</td></tr>
<tr><td><b>IVF</b></td><td>Inverted file index. A classic acceleration method for approximate nearest neighbor search (ANNS). It partitions the vector space into multiple clusters using a clustering algorithm, similar to directories. During retrieval, only a small number of the most relevant clusters are searched, significantly reducing the amount of computation. It trades a small amount of accuracy for significantly improved retrieval performance.</td></tr>
<tr><td><b>PQ</b></td><td>Product quantization. An efficient vector compression algorithm. It divides high-dimensional vectors into multiple low-dimensional subspaces and builds a separate codebook for quantization and encoding in each subspace. It can significantly reduce memory consumption, typically by tens of times, and is a core technology for vector retrieval at the billion scale and beyond.</td></tr>
<tr><td><b>SQ</b></td><td>Scalar quantization. A vector compression algorithm that reduces memory usage by independently mapping each dimension of an FP32 vector to a finite set of integers, such as INT8. Compared with PQ, SQ is simpler to implement and provides faster query performance, making it suitable for approximate retrieval scenarios that have moderate accuracy requirements</td></tr>
<tr><td><b>RaBitQ</b></td><td>Random binary quantization. An advanced, highly compressed retrieval algorithm. It uses mathematical transformations to compress FP32 vectors into 1-bit binary representations, with a theoretical compression ratio of 32x, and uses fast Hamming distance for initial filtering. It significantly reduces memory bandwidth pressure and storage costs while maintaining high recall.</td></tr>
<tr><td><b>Cagra</b></td><td>Graph-based approximate nearest neighbor search algorithm. It organizes base library vectors by constructing a neighbor graph. During retrieval, it iteratively searches along graph edges, gradually approaching the nearest neighbors. Compared with IVF-based methods, graph-based retrieval provides higher search efficiency in low-latency scenarios and is suitable for high-performance approximate retrieval with base libraries at the scale of hundreds of millions.</td></tr>
<tr><td><b>BinaryFlat</b></td><td>Binary brute-force search. An exhaustive search algorithm designed specifically for binary vectors. The vectors consist of 0s and 1s, and similarity is calculated using Hamming distance. Because it uses low-level bitwise operations (XOR), it provides extremely fast computation with very low memory usage, making it suitable for binary feature scenarios such as image fingerprinting.</td></tr>
<tr><td><b>L2</b></td><td>Euclidean distance. Measures the absolute straight-line distance between two vectors in a multidimensional space. A smaller distance indicates greater similarity between the vectors. It is suitable for scenarios that focus on absolute numerical differences, such as image pixel feature comparison.</td></tr>
<tr><td><b>IP</b></td><td>Inner product. Measures similarity by calculating the dot product of two vectors. A larger inner product indicates greater similarity. When the vectors are normalized to unit length, IP is equivalent to cosine similarity. It is widely used in scenarios that focus on directional consistency, such as text semantic matching.</td></tr>
<tr><td><b>Hamming</b></td><td>Hamming distance. Measures the difference between two binary vectors of the same length. It is calculated by counting the number of positions at which the corresponding bits differ (0 vs. 1). Fewer differing bits indicate greater similarity. It is a core metric for binary retrieval algorithms such as BinaryFlat and RaBitQ.</td></tr>
</tbody></table>

## Application Scenarios<a name="en-us_TOPIC_0000001985832242"></a>

<a name="table_scenarios"></a>
<table><tbody>
<tr><th width="25%">Question</th><th width="30%">Condition</th><th width="45%">Recommendation</th></tr>
<tr><td rowspan="2"><b>Do you need exact search results?</b></td><td>Exact results are required</td><td><b>Brute-force search algorithms</b>, which can guarantee exact results.</td></tr>
<tr><td>Minor accuracy loss is acceptable</td><td><b>Approximate nearest neighbor search algorithms</b>, which reduce memory usage and improve retrieval performance.</td></tr>
<tr><td rowspan="3"><b>How large is the base library?</b></td><td>300,000 – 1 million entries (small library)</td><td><b>AscendIndexFlat / AscendIndexSQ / AscendIndexInt8Flat</b> and other brute-force search algorithms, which provide the highest accuracy.</td></tr>
<tr><td>Tens of millions of entries (medium library)</td><td><b>AscendIndexIVFSQ / AscendIndexVStar / AscendIndexGreat</b>, which compress features to balance performance and accuracy and are suitable for medium-scale retrieval.</td></tr>
<tr><td>Hundreds of millions of entries (large library)</td><td><b>AscendIndexIVFSP / AscendIndexIVFSQT / AscendIndexIVFFlat / AscendIndexIVFPQ / AscendIndexIVFRaBitQ / AscendIndexCagra</b>, which use clustering and quantization or graph indexing to minimize memory usage and support indexing of massive datasets.</td></tr>
<tr><td rowspan="3"><b>Is device memory limited?</b></td><td>Sufficient memory</td><td><b>Brute-force search algorithms</b>, which prioritize retrieval accuracy but have the highest memory usage (except Int8Flat).</td></tr>
<tr><td>Limited memory</td><td><b>AscendIndexSQ / AscendIndexIVFFlat</b>, which sacrifice some accuracy while providing moderate memory usage.</td></tr>
<tr><td>Very limited memory</td><td>Other <b>approximate nearest neighbor search algorithms</b>, which significantly reduce memory usage and are preferred for large-scale deployment.</td></tr>
<tr><td rowspan="4"><b>What is the feature type?</b></td><td>FP32</td><td>Supports most index types and provides the broadest compatibility.</td></tr>
<tr><td>FP16</td><td><b>AscendIndexFlat / AscendIndexILFlat</b>, supporting L2 and Cos distances.</td></tr>
<tr><td>INT8</td><td><b>AscendIndexInt8Flat</b>, designed specifically for integer features and supporting L2 and Cos distances.</td></tr>
<tr><td>Binary features</td><td><b>AscendIndexBinaryFlat</b>, which uses Hamming distance for extremely fast comparison.</td></tr>
<tr><td rowspan="2"><b>Do you have other advanced feature requirements?</b></td><td>Need to filter by temporal/spatial attributes</td><td><b>AscendIndexTS</b>, which supports retrieval with multiple temporal and spatial attribute filters.</td></tr>
<tr><td>Need to search multiple libraries simultaneously</td><td>Use APIs for <b>multi-index batch search</b>.</td></tr>
</tbody></table>
