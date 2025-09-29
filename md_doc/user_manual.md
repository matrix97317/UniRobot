# User Manual

[English](./user_manual_en.md)

### 准备工作
```
git clone https://github.com/matrix97317/UniRobot.git
cd UniRobot
make dev
make pre-commit
```

### 对于模型开发人员使用流程

1. 在对应的目录下 开发特定的模型组件(例如:loss, encoder, decoder, xxx_model)

2. 注册你开发的模型组件到Slot，编辑slot_entry.json

3. 编写你的实验config文件，可以参考configs目录下demo

4. 训练模型
```
开发机(Debug机):
bash brain_spawn_run.sh -h # 查看使用说明
bash brain_spawn_run.sh DEVICE_NUM config_path EXPERIMENT_NAME

集群上:
bash brain_spawn_run.sh DEVICE_NUM config_path EXPERIMENT_NAME
```
PS: 由于集群基础环境信息不一样，请自行根据集群信息，修改UniRobot/unirobot/utils/dist_util.py中 get_node_rank()和get_nnodes()信息

### 对于机器人本体软件开发人员使用流程

1. 在对应目录下开发机器人的组件（例如： 传感器，电机，遥控器）

2. 开发机器人的控制逻辑, 可以参考 UniRobot/unirobot/robot/robot_so101.py

3. 注册你开发的组件，编辑slot_entry.json

4. 编写你的机器人控制信息config 文件，可以参考configs 目录下demo

5. 运行机器人

```
机器人本体:
pipenv run unirobot robot-run -h # 查看帮助说明
pipenv run unirobot robot-run /path/your_config 
```

