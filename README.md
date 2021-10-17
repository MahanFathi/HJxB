# HJxB
Continuous Fitted Value Iteration based on closed-form solution of Hamilton-Jacobi-Bellman equation for affine systems, implemented in JAX. This method was first used in [this paper](https://arxiv.org/abs/2105.04682) and tested on a number of classic contorl problems like cartpole or pendulum swig-up tasks. 

<img align="right" src=https://imgur.com/CePnqn3.gif style="width:400px;"/>

### This repo contains:

* Extensibility for custom environment definition
* Linearization of forward dynamics and reward functions
* A number of different solvers for bellman backup optimization
* Various integrators for forward dynamics 
* Various data collection and storage methods
* Extensive configurability 
* Tensorboard and file logging  

### Usage
For now:
``` shell
python main.py --config-file=./config/Pendulum.yaml
tensorboard --logdir=./logs/<log_dir>
```


