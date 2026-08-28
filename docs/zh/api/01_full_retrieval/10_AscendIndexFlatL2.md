# AscendIndexFlatL2<a name="ZH-CN_TOPIC_0000001456375424"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001877955534"></a>

AscendIndexFlatL2是存储FP16浮点数类型并使用L2距离的特征暴力检索算法。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

> [!NOTE]
>AscendIndexFlatL2算法支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，使用在线算子需要用户在应用程序的最后显式调用 \(void\)aclFinalize\(\) （需要包含头文件：\#include "acl/acl.h"）。

## AscendIndexFlatL2接口<a name="ZH-CN_TOPIC_0000001506495761"></a>

<a name="zh-cn_topic_0000001294312541_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatL2的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexFlatL2 *index</code></strong>：CPU侧Index资源。<br><strong><code>AscendIndexFlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为faiss::MetricType::METRIC_L2。</td></tr>
</tbody></table>

<a name="zh-cn_topic_0000001294591937_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatL2的构造函数，生成维度为dims的AscendIndexFlatL2（单个Index管理的一组向量的维度是唯一的），此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndexFlatL2管理的一组特征向量的维度。<br><strong><code>AscendIndexFlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">dims ∈ {32, 64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3072, 4096, 3584}</td></tr>
</tbody></table>

<a name="zh-cn_topic_0000001247793230_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlatL2(const AscendIndexFlatL2&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexFlatL2&amp;</code></strong>：常量AscendIndexFlatL2。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="zh-cn_topic_0000001294312453_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexFlatL2()</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlatL2的析构函数，销毁AscendIndexFlatL2对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456375400"></a>

<a name="zh-cn_topic_0000001248112146_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat基于一个已有的“index”拷贝到Ascend，清空当前的AscendIndexFlatL2底库，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexFlat *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{64, 128, 256, 384, 512, 1024, 1408, 1536, 2048, 3584}，底库向量总数的取值范围：0 &lt;= n &lt; 1e9，metric_type参数取值为faiss::MetricType::METRIC_L2。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456535052"></a>

<a name="zh-cn_topic_0000001247793178_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexFlatL2的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexFlat *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001456695116"></a>

<a name="zh-cn_topic_0000001294432513_table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlatL2&amp; operator=(const AscendIndexFlatL2&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexFlatL2&amp;</code></strong>：常量AscendIndexFlatL2。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>
