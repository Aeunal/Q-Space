# Artificial Intelligence Final Project: Q-Space 
## Ahmet Emin Ünal

# Q-Space
This repository contains the research on working with embeddings of states for RL environments with image observations (such as atari environments).

### Topics
- Training the GYM Breakout environment with a Policy Gradient method
  - REINFORCE
  - A2C
  - PPO
- Auto-encoder structure on top of the current method.
- 

### Structure

Follow the "Project.ipynb" ipython notebook for explanations: [Project](https://github.com/Aeunal/Q-Space/blob/main/Project.ipynb)

### Installation

To start to use this library, you need to install requirements. It is recommended that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment to use it.

```
conda create -n q_space python=3.7 
conda activate q_space
```

If you are going to use GPU, install [Pytorch](https://pytorch.org/get-started/locally/) using the link and remove it from requirements.

You can install the requirements with the following commands:

```
conda install -c conda-forge swig
conda install nodejs
pip install -r requirements.txt
python -m ipykernel install --user --name=q_space
```
Then you need to install the project package. You can install the package with the following command: (Make sure that you are at the project directory.)

```
pip install -e .
```

This command will install the project package in development mode so that the installation location will be the current directory.


<!-- 
### Docker

You can also use docker to work on the project. Simply build a docker image from the project directory using the following command:


```
docker build -t q_space .
```

You may need to install docker first if you don't have it already.

After building a container we need to mount the project directory at your local computer to the container we want to run. Note that the container will install necessary python packages in build.

You can run the container using the command below as long as your current directory is the project directory:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/qspace_root q_space
```

This way you can connect the container at ```localhost:8889``` in your browser. Note that, although we are using docker, changes are made in your local directory since we mounted it.

You can also use it interactively by simply running:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/qspace_root q_space /bin/bash
```

> Note: Running docker with cuda requires additional steps!

-->

### Related Readings (**Must Read**)

- Reinforcement Learning: An Introduction (2nd), Richard S. Sutton and Andrew G. Barto Chapter 12 & 13 
- [A3C](https://arxiv.org/abs/1602.01783)
- [GAE](https://arxiv.org/pdf/1506.02438.pdf)
- [PPO](https://arxiv.org/pdf/1707.06347.pdf)

### Contact
Author: Ahmet Emin Ünal
aeunal@hotmail.com
