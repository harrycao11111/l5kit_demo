# simulation

包含文件: `train.ipynb`, `simulation_test.ipynb`, `config.yaml`

算法: 同时预测所有交通参与者的轨迹，以此规划自动驾驶汽车的轨迹

## `train.ipynb`

功能: 训练非汽车的交通参与者的轨迹预测模型

## `simulation_test.ipynb`

功能：根据算法，仿真测试效果

问题：`Evaluating with simulated agents`下方代码报错，卷积层数量不匹配即输入了15通道但期望的输入通道数为5，

问题定位：问题出现在planning_model.pt上，其仅允许5个通道输入，由于./simulation没有提供planning_model.pt，因此我使用了./planning中的planning_model.pt，但两者的YAML文件不同，导致输入数据集的通道数不同。

解决办法，替换./planning/train.ipynb中使用的YAML文件为./simulation/config.yaml，训练完成后导出./sumlation/planning_model.pt，在./simulation/simulation_test.ipynb中使用它。