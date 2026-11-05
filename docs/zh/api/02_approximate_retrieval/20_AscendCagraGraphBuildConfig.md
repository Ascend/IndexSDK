# AscendCagraGraphBuildConfig

本结构体配置[AscendCagraGraphBuilder](./19_AscendCagraGraphBuilder.md)的`Build`和`BuildToFile`，定义于`faiss/ascend/AscendCagraGraphBuilder.h`，命名空间为`faiss::ascend`。

## 成员介绍

<table><tbody>
<tr><td align="center"><strong>成员</strong></td><td align="center"><strong>类型</strong></td><td align="center"><strong>默认值</strong></td><td><strong>说明与约束</strong></td></tr>
<tr><td>intermediateDegree</td><td>uint32_t</td><td>128</td><td>NN-Descent中间图度数M，须满足G≤M≤128且M&lt;N。最终图度数G在Builder::Init中设置。模型生成参数-i须与M一致。</td></tr>
<tr><td>maxIterations</td><td>uint32_t</td><td>20</td><td>最大迭代轮数，必须大于0。达到收敛阈值可提前结束；达到最大轮数不表示已收敛。</td></tr>
<tr><td>terminationThreshold</td><td>float</td><td>0.0001F</td><td>停止比例，取值[0,1]。停止更新数为floor(terminationThreshold×N×M)；本轮更新数不大于该值时结束迭代。</td></tr>
<tr><td>verbose</td><td>bool</td><td>false</td><td>是否输出构图阶段耗时、每轮更新数及收敛摘要。true时额外统计开销包含在构图时间中。</td></tr>
<tr><td>guaranteeConnectivity</td><td>bool</td><td>true</td><td>是否在Host侧执行连通性保障。开启会回传中间图并构造补边骨架，增加Host内存与耗时；常用G=64规格验证弱连通。关闭后不保证最终图连通。</td></tr>
</tbody></table>

## 配置示例

```cpp
faiss::ascend::AscendCagraGraphBuildConfig config;
config.intermediateDegree = 128;
config.maxIterations = 20;
config.terminationThreshold = 0.0001F;
config.verbose = true;
config.guaranteeConnectivity = true;
```

SIFT1M常用配置为N=1000000、D=128、M=128、G=64。配置不含TopK或检索候选数；这两项不参与构图。
修改N/D/M/G后须重新生成对应静态构图OM；修改迭代轮数、停止阈值或日志开关不改变OM形状。
