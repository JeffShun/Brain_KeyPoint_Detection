# seg_myocardialBridge
心肌桥项目
因为算法方案相同，放到一个repo实验

- 需求链接: https://wiki.infervision.com/pages/viewpage.action?pageId=321429031


# 1)下载原始数据并自动转化为训练可用的数据
make generate_data USER_NAME={用户名}

# 2)或者直接下载训练数据
make download USER_NAME={用户名}

# 3)开始训练网络模型
bash run_dist.sh -d 0,1,2,3 -g 4 -c ./config/seg_myocardialbridge_config.py

# 4)模型静态化
make save_torchscript


