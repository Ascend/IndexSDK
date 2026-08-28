# AscendIndexIVFFlat<a name="ZH-CN_TOPIC_0000002478095516"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002510095475"></a>

AscendIndexIVFFlat利用IVF进行加速，是二级近似检索算法。当前仅支持IP距离。

## AscendIndexIVFFlat接口<a name="ZH-CN_TOPIC_0000002509975505"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFFlat(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFFlatConfig config)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFFlat的构造函数，创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：底库检索向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：距离类型，当前只支持faiss::METRIC_INNER_PRODUCT。<br><strong><code>int nlist</code></strong>：IVF分桶数。<br><strong><code>AscendIndexIVFFlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims目前仅支持128。<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 32768}。</td></tr>
</tbody></table>

<a name="table663150151113"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFFlat&amp;</code></strong>：常量AscendIndexIVFFlat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendIndexIVFFlat接口<a name="ZH-CN_TOPIC_0000002477935546"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>~AscendIndexIVFFlat()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexIVFFlat的析构函数，销毁AscendIndexIVFFlat对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## operate = 接口<a name="ZH-CN_TOPIC_0000002484264062"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexIVFFlat&amp; operator=(const AscendIndexIVFFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexIVFFlat&amp;</code></strong>：常量AscendIndexIVFFlat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000002478095518"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void train(idx_t n, const float *x) override;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">对AscendIndexIVFFlat执行训练，继承AscendIndex中的相关接口并提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 训练采用k-means进行聚类，训练集比较小可能会影响查询精度。<br>● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为dims * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 当前只支持CPU聚类，不支持“useKmeansPP”参数设置为“true”。</td></tr>
</tbody></table>
