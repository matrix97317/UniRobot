
# Developer Manual



First of all, we're excited that you want to become a UniRobot developer! The following steps will guide you through the development setup process.

### STEP 0. Build Docker

First, you need to install Docker tools. For specific instructions, see: https://docs.docker.com/get-started/

If you already have a suitable Docker image, you can skip this step. If not, follow these steps:

```
$ git clone https://github.com/matrix97317/UniRobot.git
$ cd UniRobot
$ cd dockers
# If you need to develop for the Brain aspect, proceed as follows:
$ sudo docker buildx build --platform=linux/amd64 -t unirobot_brain:v1.0.0 -f BrainDockerfile .
$ sudo docker images //you can look ` unirobot_brain:v1.0.0`
# If you need to develop for the Robot aspect, proceed as follows:
$ sudo docker buildx build --platform=linux/amd64 -t unirobot_robot:v1.0.0 -f RobotDockerfile .
$ sudo docker images //you can look ` unirobot_robot:v1.0.0`
```

### STEP 1. Clone Repo
If you have already cloned the repository, you can skip this step.

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

Now you can start developing the parts that interest you (Brain OR Robot).

### STEP 4. Add your unit test.

To ensure the robustness of the project, you must provide necessary unit tests.

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
