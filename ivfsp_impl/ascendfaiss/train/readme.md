### 环境要求
arm架构
python3.7 
numpy
cann>=5.1RC2

### 相关文件描述
+ `SP.py`:IVFSP混淆后代码
+ `train.py`:命令行参数解析以及调用so里方法的py脚本
+ `train.sh`:输入训练参数执行训练的shell脚本
+ `op_models_pyacl`:训练过程中涉及的om存放文件夹，需要提前根据本次要用的参数生成对应的om


### 参数描述
+ `nCentroids`:IVF桶的个数，需对齐ascendsearch支持的IVF nList值
+ `dim`:原始特征的维度，需对齐16
+ `dim2`:码字的维度，需对齐16
+ `num_iter`:训练迭代次数
+ `device`:训练用的昇腾卡id
+ `batch_size`:训练用的batch_size，需对齐16
+ `update_codebook_batch_size`:训练用来更新码本的最大batch_size，需对齐16
+ `ratio`:训练用原始样本的采样率，默认是1.0
+ `learn_data_path`:训练用的原始特征文件路径，支持bin/npy格式，bin存储方式是行优先，数据类型是float32
+ `codebook_output_path`:训练输出码本的路径
+ `model_dir`:`op_models_pyacl`的路径






