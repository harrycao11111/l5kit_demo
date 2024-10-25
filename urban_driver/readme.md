# simulation

包含文件: `train.ipynb`, `closed_loop_test.ipynb`, `config.yaml`

算法: GNN预测自动驾驶汽车轨迹，处理输入的局部点云数据，全局特征聚合，生成预测轨迹。

## `train.ipynb`

功能: 训练非汽车的交通参与者的轨迹预测模型，输出`urban_driver.pt`

## `closed_loop_test.ipynb`

功能：调用`urban_driver.pt`，闭环测试。