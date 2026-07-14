# 背景知识<a name="ZH-CN_TOPIC_0000001985832240"></a>

## 名词解释<a name="ZH-CN_TOPIC_0000001985832241"></a>

<a name="table_glossary"></a>
<table><tbody>
<tr><th width="20%">名词</th><th width="80%">说明</th></tr>
<tr><td><b>Flat</b></td><td>暴力穷举搜索。不建立复杂的索引结构，直接将查询向量与底库中的所有向量逐一进行距离计算。其特点是召回率100%（绝对精确），但计算开销大、延迟高，常用于小规模数据集或作为其他算法的精度基准。</td></tr>
<tr><td><b>INT8</b></td><td>8位整型数值格式。一种低精度数据类型，相比标准的 FP32（32位浮点数）可减少 75% 的内存占用并提升计算吞吐量。常作为量化后的数据存储格式，用于在硬件资源受限的场景下平衡性能与精度。</td></tr>
<tr><td><b>IVF (Inverted File)</b></td><td>倒排文件索引。一种经典的近似最近邻搜索（ANNS）加速方法。通过聚类算法将向量空间划分为多个簇（类似目录），检索时仅遍历最相关的少数几个簇，从而大幅减少计算量，是以微小的精度损失换取极高的检索性能。</td></tr>
<tr><td><b>PQ (Product Quantization)</b></td><td>乘积量化。一种高效的向量深度压缩算法。它将高维向量切分为多个低维子空间，分别建立码本进行量化编码。能极大降低内存消耗（通常可压缩数十倍），是处理十亿级以上海量向量检索的核心技术。</td></tr>
<tr><td><b>SQ (Scalar Quantization)</b></td><td>标量量化。一种向量压缩算法，通过将 FP32 向量的每个维度独立映射到有限的整数集合（如 INT8）来降低内存占用。相比 PQ，SQ 实现更简单且查询速度较快，适合对精度有一定要求的近似检索场景。</td></tr>
<tr><td><b>RaBitQ (Random Binary Quantization)</b></td><td>随机二值量化。一种前沿的极致压缩检索算法。通过数学变换将 FP32 向量压缩为 1-bit 的二进制表示（理论压缩率达32倍），并利用快速汉明距离进行初筛。在保证极高召回率的同时，显著降低了内存带宽压力和存储成本。</td></tr>
<tr><td><b>Cagra</b></td><td>基于图的近似最近邻搜索算法。通过构建近邻图（Neighbor Graph）组织底库向量，检索时沿图边迭代搜索，逐步逼近最近邻。相比 IVF 类方法，图检索在低延迟场景下具有更优的搜索效率，适用于亿级底库的高性能近似检索。</td></tr>
<tr><td><b>BinaryFlat</b></td><td>二值化暴力检索。专为二进制向量设计的穷举搜索算法。向量由 0 和 1 组成，使用汉明距离计算相似度。由于底层采用位运算（XOR），其计算速度极快且内存占用极低，适用于图像指纹等二值特征场景。</td></tr>
<tr><td><b>L2 (Euclidean Distance)</b></td><td>欧几里得距离。衡量两个向量在多维空间中的绝对直线距离。距离值越小，代表两个向量越相似。适用于关注数值绝对差异的场景，如图像像素特征比对。</td></tr>
<tr><td><b>IP (Inner Product)</b></td><td>内积。通过计算两个向量的点积来衡量相似性。内积值越大，代表相似度越高。当向量经过归一化处理（模长为1）后，IP 等价于余弦相似度，广泛应用于文本语义匹配等关注方向一致性的场景。</td></tr>
<tr><td><b>Hamming (Hamming Distance)</b></td><td>汉明距离。专门用于衡量两个等长二进制向量之间的差异。通过统计对应位置上不同字符（0与1）的个数来计算距离，差异位数越少则越相似。是 BinaryFlat、RaBitQ 等二值化检索算法的核心度量标准。</td></tr>
</tbody></table>

## 应用场景<a name="ZH-CN_TOPIC_0000001985832242"></a>

<a name="table_scenarios"></a>
<table><tbody>
<tr><th width="25%">问题</th><th width="30%">条件</th><th width="45%">选择建议</th></tr>
<tr><td rowspan="2"><b>是否需要精确搜索结果？</b></td><td>需要完全精确的结果</td><td><b>全量检索算法</b>，能保证完全精确结果的索引类型。</td></tr>
<tr><td>可接受少量精度损失</td><td><b>近似检索算法</b>，能减少内存占用，并提升检索性能。</td></tr>
<tr><td rowspan="3"><b>底库规模有多大？</b></td><td>30万~100万条（小库）</td><td><b>AscendIndexFlat / AscendIndexSQ / AscendIndexInt8Flat</b> 等全量检索算法，精度最高。</td></tr>
<tr><td>千万级（中库）</td><td><b>AscendIndexIVFSQ / AscendIndexVStar / AscendIndexGreat</b>，压缩特征，平衡性能与精度，适合中等规模检索。</td></tr>
<tr><td>亿级（大库）</td><td><b>AscendIndexIVFSP / AscendIndexIVFSQT / AscendIndexIVFFlat / AscendIndexIVFPQ / AscendIndexIVFRaBitQ / AscendIndexCagra</b>，聚类+量化或图索引，极致压缩内存，支撑海量数据索引。</td></tr>
<tr><td rowspan="3"><b>内存（Device内存）是否受限？</b></td><td>内存充足</td><td><b>全量检索算法</b>，优先保障检索精度，但内存占用最大（Int8Flat除外）。</td></tr>
<tr><td>内存受限</td><td><b>AscendIndexSQ / AscendIndexIVFFlat</b>，损失一定精度，内存占用中等。</td></tr>
<tr><td>内存非常受限</td><td>其他 <b>近似检索算法</b>，大幅降低内存占用，大规模部署首选。</td></tr>
<tr><td rowspan="4"><b>特征类型是什么？</b></td><td>FP32</td><td>支持大部分索引类型，通用性最强。</td></tr>
<tr><td>FP16</td><td><b>AscendIndexFlat / AscendIndexILFlat</b>，支持L2和Cos距离。</td></tr>
<tr><td>INT8</td><td><b>AscendIndexInt8Flat</b>，专为整型特征设计，支持L2和Cos距离。</td></tr>
<tr><td>二值化特征</td><td><b>AscendIndexBinaryFlat</b>，使用汉明距离计算，进行极速对比。</td></tr>
<tr><td rowspan="2"><b>是否有其他高级功能需求？</b></td><td>需要按时间/空间等属性过滤</td><td><b>AscendIndexTS</b>，支持时间空间多属性过滤检索。</td></tr>
<tr><td>需要从多个库同时检索</td><td>使用 <b>多Index批量检索</b> 相关接口。</td></tr>
</tbody></table>
