# Background Knowledge<a name="ZH-CN_TOPIC_0000001985832240"></a>

## Glossary<a name="ZH-CN_TOPIC_0000001985832241"></a>

<a name="table_glossary"></a>
<table><tbody>
<tr><th width="20%">Term</th><th width="80%">Description</th></tr>
<tr><td><b>Flat</b></td><td>Brute-force search. No complex index structure is built. The query vector is compared against all vectors in the base library one by one to compute distances. It features 100% recall (absolutely precise), but has high computational overhead and latency. It is commonly used for small-scale datasets or as an accuracy baseline for other algorithms.</td></tr>
<tr><td><b>INT8</b></td><td>8-bit integer format. A low-precision data type that reduces memory usage by 75% compared to standard FP32 (32-bit floating point) and improves computational throughput. It is commonly used as a quantized data storage format to balance performance and precision in hardware-constrained scenarios.</td></tr>
<tr><td><b>IVF (Inverted File)</b></td><td>Inverted file index. A classic acceleration method for approximate nearest neighbor search (ANNS). It partitions the vector space into multiple clusters using a clustering algorithm (similar to a directory). During retrieval, only the most relevant clusters are traversed, significantly reducing computation. It trades a small amount of precision for greatly improved retrieval performance.</td></tr>
<tr><td><b>PQ (Product Quantization)</b></td><td>Product quantization. An efficient deep vector compression algorithm. It splits high-dimensional vectors into multiple low-dimensional subspaces and builds codebooks for quantization encoding in each subspace. It can greatly reduce memory consumption (typically by tens of times) and is a core technology for handling vector retrieval at the billion-scale and beyond.</td></tr>
<tr><td><b>SQ (Scalar Quantization)</b></td><td>Scalar quantization. A vector compression algorithm that reduces memory usage by independently mapping each dimension of an FP32 vector to a finite set of integers (such as INT8). Compared to PQ, SQ is simpler to implement and offers faster query speeds, making it suitable for approximate retrieval scenarios that require a certain level of precision.</td></tr>
<tr><td><b>RaBitQ (Random Binary Quantization)</b></td><td>Random binary quantization. A cutting-edge extreme compression retrieval algorithm. It compresses FP32 vectors into 1-bit binary representations through mathematical transformations (theoretical compression ratio of 32x) and uses fast Hamming distance for initial screening. It significantly reduces memory bandwidth pressure and storage costs while maintaining high recall rates.</td></tr>
<tr><td><b>Cagra</b></td><td>Graph-based approximate nearest neighbor search algorithm. It organizes base library vectors by constructing a neighbor graph. During retrieval, it iteratively searches along graph edges, gradually approaching the nearest neighbors. Compared to IVF-based methods, graph retrieval offers better search efficiency in low-latency scenarios and is suitable for high-performance approximate retrieval with billion-level base libraries.</td></tr>
<tr><td><b>BinaryFlat</b></td><td>Binary brute-force retrieval. An exhaustive search algorithm designed specifically for binary vectors. Vectors consist of 0s and 1s, and similarity is calculated using Hamming distance. Because it uses bitwise operations (XOR) at the low level, it is extremely fast and has very low memory usage, making it suitable for binary feature scenarios such as image fingerprinting.</td></tr>
<tr><td><b>L2 (Euclidean Distance)</b></td><td>Euclidean distance. Measures the absolute straight-line distance between two vectors in a multi-dimensional space. A smaller distance value indicates that the two vectors are more similar. It is suitable for scenarios that focus on absolute numerical differences, such as image pixel feature comparison.</td></tr>
<tr><td><b>IP (Inner Product)</b></td><td>Inner product. Measures similarity by computing the dot product of two vectors. A larger inner product value indicates higher similarity. When vectors are normalized (unit length), IP is equivalent to cosine similarity. It is widely used in scenarios that focus on directional consistency, such as text semantic matching.</td></tr>
<tr><td><b>Hamming (Hamming Distance)</b></td><td>Hamming distance. Specifically used to measure the difference between two equal-length binary vectors. It calculates the distance by counting the number of positions where the corresponding characters differ (0 vs. 1). Fewer differing bits indicate greater similarity. It is the core metric for binary retrieval algorithms such as BinaryFlat and RaBitQ.</td></tr>
</tbody></table>

## Application Scenarios<a name="ZH-CN_TOPIC_0000001985832242"></a>

<a name="table_scenarios"></a>
<table><tbody>
<tr><th width="25%">Question</th><th width="30%">Condition</th><th width="45%">Recommendation</th></tr>
<tr><td rowspan="2"><b>Do you need exact search results?</b></td><td>Exact results are required</td><td><b>Full retrieval algorithms</b>, index types that guarantee completely exact results.</td></tr>
<tr><td>Minor precision loss is acceptable</td><td><b>Approximate retrieval algorithms</b>, which reduce memory usage and improve retrieval performance.</td></tr>
<tr><td rowspan="3"><b>How large is the base library?</b></td><td>300K–1M records (small library)</td><td><b>AscendIndexFlat / AscendIndexSQ / AscendIndexInt8Flat</b> and other full retrieval algorithms, offering the highest precision.</td></tr>
<tr><td>Tens of millions (medium library)</td><td><b>AscendIndexIVFSQ / AscendIndexVStar / AscendIndexGreat</b>, compressing features to balance performance and precision, suitable for medium-scale retrieval.</td></tr>
<tr><td>Hundreds of millions (large library)</td><td><b>AscendIndexIVFSP / AscendIndexIVFSQT / AscendIndexIVFFlat / AscendIndexIVFPQ / AscendIndexIVFRaBitQ / AscendIndexCagra</b>, using clustering + quantization or graph indexing for extreme memory compression, supporting massive data indexing.</td></tr>
<tr><td rowspan="3"><b>Is device memory limited?</b></td><td>Sufficient memory</td><td><b>Full retrieval algorithms</b>, prioritizing retrieval precision, but with the highest memory usage (except Int8Flat).</td></tr>
<tr><td>Limited memory</td><td><b>AscendIndexSQ / AscendIndexIVFFlat</b>, sacrificing some precision with moderate memory usage.</td></tr>
<tr><td>Very limited memory</td><td>Other <b>approximate retrieval algorithms</b>, significantly reducing memory usage, preferred for large-scale deployment.</td></tr>
<tr><td rowspan="4"><b>What is the feature type?</b></td><td>FP32</td><td>Supports most index types, offering the broadest compatibility.</td></tr>
<tr><td>FP16</td><td><b>AscendIndexFlat / AscendIndexILFlat</b>, supporting L2 and Cos distances.</td></tr>
<tr><td>INT8</td><td><b>AscendIndexInt8Flat</b>, designed for integer features, supporting L2 and Cos distances.</td></tr>
<tr><td>Binary features</td><td><b>AscendIndexBinaryFlat</b>, using Hamming distance for ultra-fast comparison.</td></tr>
<tr><td rowspan="2"><b>Any advanced feature requirements?</b></td><td>Need to filter by time/spatial attributes</td><td><b>AscendIndexTS</b>, supporting multi-attribute filtered retrieval with time and spatial constraints.</td></tr>
<tr><td>Need to search across multiple libraries simultaneously</td><td>Use <b>multi-Index batch retrieval</b> related interfaces.</td></tr>
</tbody></table>
