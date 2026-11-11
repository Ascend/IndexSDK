# AscendIndexFlatConfig<a name="ZH-CN_TOPIC_0000001456375216"></a>

AscendIndexFlat需要使用对应的AscendIndexFlatConfig执行对应资源的初始化。

**接口说明<a name="section140920164419"></a>**

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatConfig的默认构造函数。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table46951722104415"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::initializer_list&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM, bool isShard = true)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::initializer_list&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“FLAT_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_L2”时建议设置1024MB。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_INNER_PRODUCT”时建议设置1280MB。<br><strong><code>bool isShard</code></strong>：分片开关，默认true。true将底库分片到各设备；false在每个设备保存完整副本，并轮询选择一个设备执行每次查询调用。对应公开成员<code>shard</code>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

<a name="table842319354444"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>inline AscendIndexFlatConfig(std::vector&lt;int&gt; devices, int64_t resourceSize = FLAT_DEFAULT_MEM, bool isShard = true)</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatConfig的构造函数，生成AscendIndexFlatConfig，此时根据“devices”中配置的值设置Device侧昇腾AI处理器资源，配置资源池大小。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>std::vector&lt;int&gt; devices</code></strong>：Device侧设备ID。<br><strong><code>int64_t resourceSize</code></strong>：设备侧预置的内存池大小，单位为Byte，计算过程中存储中间结果的内存空间，用于避免计算过程中动态申请内存造成性能波动。默认参数为头文件中的“FLAT_DEFAULT_MEM”。该参数通过底库大小和search的batch数共同确定，在底库大于或等于4194304且batch数大于或等于16时参考以下建议进行设置。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_L2”时建议设置1024MB。<br>● 当AscendIndexFlat的距离类型为“faiss::METRIC_INNER_PRODUCT”时建议设置1280MB。<br><strong><code>bool isShard</code></strong>：分片开关，默认true。true将底库分片到各设备；false在每个设备保存完整副本，并轮询选择一个设备执行每次查询调用。对应公开成员<code>shard</code>。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● “devices”需要为合法有效不重复的设备ID，最大数量为64。<br>● “resourceSize”配置的值不超过10 * 1024MB（10 * 1024 * 1024 * 1024字节），当设置为“-1”时，Device侧昇腾AI处理器资源配置为默认值128MB。</td></tr>
</tbody></table>

## 分片与副本模式<a name="flat-shard-mode"></a>

公开成员 `bool shard = true` 也可在构造索引前设置。`shard=true` 保持原有分片行为；`shard=false` 时，每个设备均需容纳完整底库和查询资源。配置传入索引后，不支持通过修改原配置对象切换索引模式。

- 新增配置成员改变结构布局，升级后应用与IndexSDK库应配套重新编译；默认参数保持原有构造调用方式的源码兼容。
- 单设备可使用任一模式。多设备 `search_with_selector` 仅支持 `shard=false`。
- 多设备副本在索引构造时检查既有多线程安全开关，必须在启动前设置 `export MX_INDEX_MULTITHREAD=1`，即使当前调用方暂时仅使用一个线程。
- 该环境变量并非本次新增；它控制索引锁、线程安全内存栈和算子管理。首次读取后缓存，初始化后再设置不会切换模式。
- 开关不会自动创建业务查询线程。需由调用方使用固定工作线程并发查询，每次调用将完整batch交给一个副本，不进行跨设备TopK合并。
- 添加、删除和重置同步作用于副本；`ntotal` 和删除返回数量表示逻辑向量数，`copyTo` 只导出一份底库。

```cpp
faiss::ascend::AscendIndexFlatConfig config(
    {0, 1}, faiss::ascend::FLAT_DEFAULT_MEM, false);
```

完整示例见[AscendIndexFlat的Flat Bitmap过滤检索](./08_AscendIndexFlat.md#flat-bitmap-filtering)。
