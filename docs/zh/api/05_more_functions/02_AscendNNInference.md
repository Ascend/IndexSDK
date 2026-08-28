# AscendNNInference<a name="ZH-CN_TOPIC_0000001456375320"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456535204"></a>

通过神经网络执行推理。

## AscendNNInference接口<a name="ZH-CN_TOPIC_0000001456854780"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendNNInference(std::vector&lt;int&gt; deviceList, const char* model, uint64_t modelSize);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendNNInference的构造函数，生成AscendNNInference，此时根据“deviceList”中配置的值设置Device侧昇腾AI处理器资源以及模型路径等。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; deviceList</code></strong>：Device侧设备ID。<br><strong><code>const char* model</code></strong>：深度神经网络推理模型。<br><strong><code>uint64_t modelSize</code></strong>：深度神经网络推理模型的大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● deviceList取值范围(0, 32]。<br>● “model”需要为合法有效的深度神经网络推理模型的内存指针，大小为“modelSize”，modelSize取值范围为(0, 128MB]，参数不匹配可能造成模型实例化或推理失败。非法的模型可能会对系统造成危害，请确保模型的来源合法有效。dimsIn ∈ {64, 128, 256, 384, 512, 768, 1024}。<br>● dimsOut ∈ {32, 64, 96, 128, 256}。<br>● batches ∈ {1, 2, 4, 8, 16, 32, 64, 128}</td></tr>
</tbody></table>

<a name="table1246213101873"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendNNInference(const AscendNNInference&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此AscendNNInference拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendNNInference&amp;</code></strong>：常量AscendNNInference。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## \~AscendNNInference接口<a name="ZH-CN_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>~AscendNNInference();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendNNInference的析构函数，销毁AscendNNInference对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getDimBatch接口<a name="ZH-CN_TOPIC_0000001506334797"></a>

<a name="zh-cn_topic_0000001287392566_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getDimBatch() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取模型的单次推理的样本或查询向量的数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">模型单次推理的样本或查询向量的数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getInputType接口<a name="ZH-CN_TOPIC_0000001456854776"></a>

<a name="zh-cn_topic_0000001340072289_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getInputType() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取模型的输入数据类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">模型的输入数据类型。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getOutputType接口<a name="ZH-CN_TOPIC_0000001456854868"></a>

<a name="zh-cn_topic_0000001340232437_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getOutputType() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取模型的输出数据类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">模型的输出数据类型。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getDimIn接口<a name="ZH-CN_TOPIC_0000001456535128"></a>

<a name="zh-cn_topic_0000001287712442_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getDimIn() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取模型的输入数据维度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">输入数据维度。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## getDimOut接口<a name="ZH-CN_TOPIC_0000001456695056"></a>

<a name="zh-cn_topic_0000001287552486_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>int getDimOut() const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取模型的输出数据维度。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">模型的输出数据维度。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## infer接口<a name="ZH-CN_TOPIC_0000001506495709"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void infer(size_t n, const char* inputData, char* outputData) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据网络模型执行推理。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>size_t n</code></strong>：待执行推理的输入数量。<br><strong><code>const char* inputData</code></strong>：待执行推理的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>char* outputData</code></strong>：执行推理得到的特征向量结果。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处指针“inputData”需要为非空指针，且长度应该为dimIn * <strong><code>n</code></strong>，“outputData”需要为非空指针，且长度应该为dimOut * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456535156"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendNNInference&amp; operator=(const AscendNNInference&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendNNInference&amp;</code></strong>：常量AscendNNInference。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
