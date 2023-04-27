## 文件结构说明

```
├── train
│    └── config
│        ├── seg_myocardialbridge_config.py
│    └── custom
│        ├── datastet
│            ├── dataset.py
│            └── __init__.py
│        ├── model
│            ├── segmyocardialbridge_head.py
│            └── segmyocardialbridge_network.py
│            └── __init__.py
│        ├── utils
│            ├── generate_dataset.py
│            ├── save_torchscript.py
│            └── __init__.py
│        ├── __init__.py
│    └── train.py
│    └── run_dist.sh
│    └── requirements.txt
│    └── Makefile
│    └── README.md
├── README.md
```


- train.py: 训练代码入口，需要注意的是，在train.py里import custom，训练相关需要注册的模块可以直接放入到custom文件夹下面，会自动进行注册; 一般来说，训练相关的代码务必放入到custom文件夹下面!!!<br>

- ./custom/dataset/dataset.py: dataset类，需要@DATASETS.register_module进行注册方可被识别<br>

- ./custom/model/segmyocardialbridge_network.py: 模型head文件，需要@HEADS.register_module进行注册方可被识别<br>

- ./custom/model/segmyocardialbridge_head.py: 整个网络部分，训练阶段构建模型，forward方法输出loss的dict, 通过@NETWORKS.register_module进行注册

- ./custom/utils/generate_dataset.py: 从原始数据生成输入到模型的数据，供custom/dataset/dataset.py使用

- ./custom/utils/save_torchscript.py: 生成模型对应的静态图(根据SegMyocardialbridge_Network中的forward_test函数)

- ./config/seg_myocardialbridge_config.py: 训练的配置文件

- run_dist.sh: 分布式训练的运行脚本
