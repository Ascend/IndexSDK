# IReduction<a name="ZH-CN_TOPIC_0000001456694992"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506615161"></a>

IReduction是特征检索组件中降维方法的统一接口，目前支持**PCAR**和**NN**两种降维算法。

## CreateReduction接口<a name="ZH-CN_TOPIC_0000001456695108"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>IReduction *CreateReduction(std::string typeName, const ReductionConfig &amp;config);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">创建具体的降维算法。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::string typeName</code></strong>：降维算法参数，可选{&quot;NN&quot;, &quot;PCAR&quot;}。<br><strong><code>ReductionConfig &amp;config</code></strong>：降维参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle"><strong><code>IReduction *CreateReduction</code></strong>：创建的具体的降维实例。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">目前仅支持使用NN、PCAR两种降维参数，使用其他参数降维会抛异常。<br>使用完毕该实例后请注意delete此指针，释放对应的空间。</td></tr>
</tbody></table>

## reduce接口<a name="ZH-CN_TOPIC_0000001456375280"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void reduce(idx_t n, const float *x, float *res) const = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">降维接口，本函数中不提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：待执行推理的输入数量。<br><strong><code>const float *x</code></strong>：待执行推理的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float* res</code></strong>：执行推理得到的特征向量结果。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为dimIn * <strong><code>n</code></strong>，“res”需要为非空指针，且长度应该为dimOut * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## ReductionConfig接口<a name="ZH-CN_TOPIC_0000001456375264"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">成员</td><td valign="middle">类型</td><td valign="middle">说明</td></tr>
<tr><td width="140" align="center" valign="middle">dimIn</td><td valign="middle">int</td><td valign="middle">输入特征维度，即降维前的维度。PCAR需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">dimOut</td><td valign="middle">int</td><td valign="middle">输出特征维度，即降维后的维度。PCAR需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">eigenPower</td><td valign="middle">float</td><td valign="middle">奇异值的power数。PCAR需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">randomRotation</td><td valign="middle">bool</td><td valign="middle">是否进行随机旋转。PCAR需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">deviceList</td><td valign="middle">std::vector\&lt;int&gt;</td><td valign="middle">Device侧资源配置。NN需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">model</td><td valign="middle">const char *</td><td valign="middle">神经网络降维模型。NN需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">modelSize</td><td valign="middle">uint64_t</td><td valign="middle">模型的大小。NN需要配置此参数。</td></tr>
</tbody></table>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline ReductionConfig(int dimIn, int dimOut, float eigenPower, bool randomRotation);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">ReductionConfig的构造函数，当用户使用“PCAR”降维时，使用该函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dimIn</code></strong>：输入特征维度，即降维前的维度，PCAR需要配置此参数。<br><strong><code>int dimOut</code></strong>：输出特征维度，即降维后的维度，PCAR需要配置此参数。<br><strong><code>float eigenPower</code></strong>：奇异值的power数，PCAR需要配置此参数。<br><strong><code>bool randomRotation</code></strong>：是否进行随机旋转，PCAR需要配置此参数。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 使用不同的降维算法，需要配置对应的参数并且降维后的维度需要满足后续使用降维数据Index的维度限制。<br>● 使用PCAR降维时，需要保证dimOut&gt;0，dimIn ≥ dimOut。<strong><code>eigenPower</code></strong>的范围为[-0.5, 0]。</td></tr>
</tbody></table>

<a name="table2034112619"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline ReductionConfig(std::vector&lt;int&gt; deviceList, const char *model, uint64_t modelSize);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">ReductionConfig的构造函数，当用户使用“NN”降维时，使用该函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; deviceList</code></strong>：Device侧资源配置。<br><strong><code>const char *model</code></strong>：神经网络降维模型。<br><strong><code>uint64_t modelSize</code></strong>：模型的大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● deviceList取值范围(0, 32]。<br>● 使用不同的降维算法，需要配置对应的参数并且降维后的维度需要满足后续使用降维数据Index的维度限制。<br>● “model”需要为合法有效的深度神经网络推理模型的内存指针，大小为“modelSize”，modelSize取值范围为(0, 128MB]，参数不匹配可能造成模型实例化或推理失败。非法的模型可能会对系统造成危害，请确保模型的来源合法有效。dimsIn ∈ {64, 128, 256, 384, 512, 768, 1024}。<br>● dimsOut ∈ {32, 64, 96, 128, 256}。<br>● batches ∈ {1, 2, 4, 8, 16, 32, 64, 128}</td></tr>
</tbody></table>

## \~IReduction接口<a name="ZH-CN_TOPIC_0000001714244661"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~IReduction() = default;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">IReduction的析构函数，销毁IReduction对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## train接口<a name="ZH-CN_TOPIC_0000001506495753"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual void train(idx_t n, const float *x) const = 0;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">训练的抽象接口，本函数中不提供具体实现。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：训练集中特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“x”需要为非空指针，且长度应该为dimIn * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>
