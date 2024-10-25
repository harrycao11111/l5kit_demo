# safepathnet

包含文件: `agent_prediction.ipynb`, `config.yaml`

算法: 即论文[Safe Real-World Autonomous Driving by Learning to Predict and Plan with a Mixture of Experts](https://arxiv.org/abs/2211.02131)，Transformer处理输入数据(如地图信息、道路参与者状态、动态障碍等), Mixture of Experts方预测未来的多种行驶轨迹，选择最安全的轨迹，

## `agent_prediction.ipynb`

功能: 实现以上算法并输出`pred.csv`

注：pred.csv过大故删除