## 文件结构说明

- train.py: 单卡训练代码入口

- train_multi_gpu.py: 分布式训练代码入口

- ./custom/dataset/dataset.py: dataset类

- ./custom/model/model_network.py: 模型head文件

- ./custom/model/model_head.py: 整个网络部分，训练阶段构建模型，forward方法输出loss的dict

- ./custom/utils/generate_dataset.py: 从原始数据生成输入到模型的数据，供custom/dataset/dataset.py使用

- ./custom/utils/save_torchscript.py: 生成模型对应的静态图

- ./custom/utils/common_tools.py: 工具函数包

- ./custom/utils/distributed_utils.py: 分布式训练函数包

- ./config/detect_keypoint_config.py: 训练的配置文件
