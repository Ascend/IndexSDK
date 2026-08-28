# AscendCloner<a name="ZH-CN_TOPIC_0000001506334577"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001456375412"></a>

Index SDK提供了将NPU上的检索Index资源拷贝到CPU侧Faiss的操作，拷贝过程发生在内存中，原始NPU的Index上加载的数据会被拷贝到CPU侧的内存中，方便用户在CPU上使用相同的底库执行检索。

> [!NOTE]
>部分版本的Faiss中提供了将内存中的Index落盘（内存中的数据保存到本地硬盘）的方法，用户在基于Index SDK和Faiss处理某些敏感数据时需要特别注意提供对应的权限控制和加密保护。

## index\_ascend\_to\_cpu接口<a name="ZH-CN_TOPIC_0000001506334821"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据Ascend上的检索index资源，拷贝生成一个CPU上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::Index *ascend_index</code></strong>：Ascend上的Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">生成一个CPU上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">使用完毕该接口返回的Index指针后请注意delete掉此指针，释放对应的空间。</td></tr>
</tbody></table>

## index\_cpu\_to\_ascend接口<a name="ZH-CN_TOPIC_0000001456695032"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>faiss::Index *index_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据CPU上的检索Index资源，拷贝生成一个Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：NPU上待配置的设备ID。<br><strong><code>const faiss::Index *index</code></strong>：CPU上的检索Index资源。<br><strong><code>const AscendClonerOptions *options = nullptr</code></strong>：待配置的AscendClonerOptions资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">生成一个Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 使用完毕该接口返回的Index指针后请注意delete掉此指针，释放对应的空间。<br>● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “index”需要为合法有效的CPU Index指针。</td></tr>
</tbody></table>

<a name="table22143401019"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>faiss::Index *index_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据CPU上的检索Index资源，拷贝生成一个Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：NPU上待配置的设备ID。<br><strong><code>const faiss::Index *index</code></strong>：CPU上的检索Index资源。<br><strong><code>const AscendClonerOptions *options = nullptr</code></strong>：待配置的AscendClonerOptions资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">生成一个Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 使用完毕该接口返回的Index指针后请注意delete掉此指针，释放对应的空间。<br>● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “index”需要为合法有效的CPU Index指针。</td></tr>
</tbody></table>

## index\_int8\_ascend\_to\_cpu接口<a name="ZH-CN_TOPIC_0000001506414761"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据Ascend上的INT8的检索Index资源，拷贝生成一个CPU上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexInt8 *index</code></strong>：Ascend上的Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">生成一个CPU上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 使用完毕该接口返回的Index指针后请注意delete此指针，释放对应的空间。<br>● “index”需要为合法有效的AscendIndexInt8指针。</td></tr>
</tbody></table>

## index\_int8\_cpu\_to\_ascend接口<a name="ZH-CN_TOPIC_0000001456375248"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据CPU上的检索Index资源，拷贝生成一个Ascend上的INT8的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：NPU上待配置的设备ID。<br><strong><code>const faiss::Index *index</code></strong>：CPU上的检索Index资源。<br><strong><code>const AscendClonerOptions *options = nullptr</code></strong>：待配置的AscendClonerOptions资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">生成一个Ascend上的INT8的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 使用完毕该接口返回的Index指针后请注意delete此指针，释放对应的空间。<br>● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “index”需要为合法有效的CPU Index指针。</td></tr>
</tbody></table>

<a name="table161071151116"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">根据CPU上的检索Index资源，拷贝生成一个Ascend上的INT8的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：NPU上待配置的设备ID。<br><strong><code>const faiss::Index *index</code></strong>：CPU上的检索Index资源。<br><strong><code>const AscendClonerOptions *options = nullptr</code></strong>：待配置的AscendClonerOptions资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">生成一个Ascend上的INT8的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 使用完毕该接口返回的Index指针后请注意delete此指针，释放对应的空间。<br>● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “index”需要为合法有效的CPU Index指针。</td></tr>
</tbody></table>
