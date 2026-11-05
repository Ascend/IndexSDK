# CagraRabitq

CAGRA 图检索算子，使用 RaBitQ 量化距离和 bitonic 排序，运行于 Ascend 950。

| 参数 | 类型 | 形状与含义 |
| --- | --- | --- |
| query | FP32 | [Q,D]，原始查询 |
| knn_graph | uint32 | [N,G]，底库近邻图 |
| visited_hashmap_ptr | uint32 | [64,ceil(N/32)]，每次查询由 kernel 清零 |
| ptr | FP32 | [N,D]，底库 |
| precompute_all | FP32 | [4Q]，查询量化预计算量 |
| code | uint8 | [N×(6ceil(D/8)+16)]，底库编码与预计算量 |
| rotated_qq_ptr_all | uint8 | [Q×6ceil(D/8)]，查询位平面编码 |
| result_distances_ptr | FP32 | [Q,K]，近似距离 |
| result_indices_ptr | uint32 | [Q,K]，底库行号 |
| top_k | int 属性 | K，须与输出宽度一致 |

- 单个算子 Q 为 1/2/4/8/16/32/64；SDK 对较大查询批次自动分页。
- D 支持 {64,128,256,512,1024,1536,3072}，G 支持 {64,128,256,512}。
- 1≤K≤min(256,N)，内部候选容量为 next_power_of_two(max(K,64))。
- TopK 大于 256 时，现有更大容量排序与本地内存组合尚未完成验证，当前不开放。
- 输入采用 ND 格式；查询码、底库码必须来自同一底库的量化变换。
- 未命中支持的tiling key时触发设备异常，使算子执行失败。

安装算子包后，用 `cagra_generate_model.py -data_base N -d D -degree G -topK K -t Ascend950PR` 生成七档查询 OM。SDK 的 Init、Search 与模型中的 top_k 必须一致。更新此实现后，须重编 Host 库、安装算子包并重新生成检索 OM。
