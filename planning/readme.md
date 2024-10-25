# agent_motion_prediction

包含文件: `train.ipynb`, `open_loop_teat.ipynb`, `closed_loop_test.ipynb`, `config.yaml`

算法: DNN

## `train.ipynb`

功能: 将鸟瞰图视角的车辆轨迹作为数据集，添加了轨迹扰动，训练模型，并保存模型参数至`planning_model.pt`

## `open_loop_teat.ipynb`

功能: 开环测试数据集效果，根据车辆现有行驶状态实时预测车辆未来轨迹，但**仅仅是预测**，并与真实轨迹进行对比，指标包括：ADE、FDE

## `closed_loop_test.ipynb`

功能：闭环测试数据集效果，根据车辆现有行驶状态预测车辆未来轨迹，并**控制车辆沿着这条轨迹连续行驶**，并于真实轨迹进行对比，指标包括：ADE、FDE