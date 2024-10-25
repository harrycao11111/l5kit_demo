# agent_motion_prediction

包含文件: `drivenet_train.py`, `drivenet_eval.py`, `drivenet_config.yaml`

**存在的问题: 使用`pip install -e ."[dev]"`导入的库缺失`stable_baselines3`库，需要自行下载，但是最新版的`stable_baselines3`有依赖项numpy=1.24.0，与ly5kit的依赖项numpy=1.15.0冲突，需要找到一个合适的不冲突的版本**

**解决方法：使用`pip install -e ."[dev]"`，然后`pip install stable_baselines3== --no-deps`，接着根据`pip check`找到缺失的依赖项，最后根据现有依赖项的版本安装适合版本的库，如`pip install scipy==1.5.4`等，详情见`requirments.txt`**

算法: 

## `drivenet_train.py`

**存在的问题: 一堆，如开头缺少数据路径，YAML配置错误等**

**解决方法：修正内容，并将代码修改至drivenet_train.ipynb**

功能：训练预测车辆以外的交通参与者的轨迹

## `drivenet_eval.py`

**存在的问题: 同上**

**解决方法：同上**

这个跑得很慢，还要跑很久，所以就不跑了

功能：通过预测所以交通参与者的轨迹，规划自动驾驶车辆的轨迹

