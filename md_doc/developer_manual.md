
# Developer Manual

首先，很开心你想成为一名UniRobot 开发者，下面将一步一步引导你完成开发流程设置。

### STEP 0. Build Docker

首先需要你安装Docker工具，具体操作参见:https://docs.docker.com/get-started/

如果你已经有合理的Docker镜像了，你可以跳过该步骤。如果没有，可以按照以下流程操作:

```
$ git clone https://github.com/matrix97317/UniRobot.git
$ cd UniRobot
$ cd dockers
# 如果你是需要进行Brain方面的开发则进行如下操作:
$ sudo docker buildx build --platform=linux/amd64 -t unirobot_brain:v1.0.0 -f BrainDockerfile .
$ sudo docker images //you can look ` unirobot_brain:v1.0.0`
# 如果你是需要进行Robot方面的开发则进行如下操作:
$ sudo docker buildx build --platform=linux/amd64 -t unirobot_robot:v1.0.0 -f RobotDockerfile .
$ sudo docker images //you can look ` unirobot_robot:v1.0.0`
```

### STEP 1. Clone Repo
如果你已经Clone该Repo可以跳过该步骤

```
$ git clone https://github.com/matrix97317/UniRobot.git
$ cd UniRobot
$ git checkout -b <your_name>/<feature_name>
```

### STEP 2. Build Development Environment

```
$ cd UniRobot
$ make dev
$ make pre-commit
```

### STEP 3. Develop Project

现在你可以开始开发你感兴趣的部分了（Brain OR Robot）

### STEP 4. Add your unit test.

为了保证项目的健壮性，你必须提供必要的单元测试。

### STEP 5. Upload your commit to your branch

Congratulations！

```
$ make test       // Verify Unit Test
$ make pre-commit // Verify Code Style & auto format your code.

$ git checkout -b <your_name>/<feature_name>
$ git add <modified_file_names> # Add the intended files to commit.
$ git commit -m "commit message"
$ git checkout main
$ git pull
$ git checkout -b <your_name>/<feature_name>
$ git merge main
$ git push
```
