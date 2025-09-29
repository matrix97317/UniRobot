# User Manual



### Preparations
```
git clone https://github.com/matrix97317/UniRobot.git
cd UniRobot
make dev
make pre-commit
```

### Usage Process for Model Developers

1. Develop specific model components (e.g., loss, encoder, decoder, xxx_model) in the corresponding directories.

2. Register your developed model components to the Slot by editing slot_entry.json.

3. Write your experiment configuration file. You can refer to the demo in the configs directory.

4. Train the model:
```
Development Machine (Debug Machine):
bash brain_spawn_run.sh -h # View usage instructions
bash brain_spawn_run.sh DEVICE_NUM config_path EXPERIMENT_NAME

On the cluster:
bash brain_spawn_run.sh DEVICE_NUM config_path EXPERIMENT_NAME
```
PS: Due to differences in cluster base environment information, please modify the get_node_rank() and get_nnodes() information in UniRobot/unirobot/utils/dist_util.py according to your cluster information.

### Usage Process for Robot Software Developers

1. Develop robot components (e.g., sensors, motors, remote controllers) in the corresponding directories.

2. Develop the robot's control logic. You can refer to UniRobot/unirobot/robot/robot_so101.py.

3. Register your developed components by editing slot_entry.json.

4. Write your robot control configuration file. You can refer to the demo in the configs directory.

5. Run the robot:
```
On the robot body:
pipenv run unirobot robot-run -h # View help instructions
pipenv run unirobot robot-run /path/your_config 
```

