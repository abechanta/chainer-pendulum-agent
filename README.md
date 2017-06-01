# chainer pendulum agent

Experimental DQN implementation with Chainer for OpenAI Gym classic control environment "Pendulum-v0".

See below how it works:
- https://gym.openai.com/evaluations/eval_sMwo0KrXSV2I16456yveFw

## Usage

### Training

To train your agent, type below:

```
python run.py --train --episode 300
```

This will iterate 300 episodes for training Action-Value function (Q function) and store trained model to './model.trained/' folder.

To train more, simply type same command.
Trained model will be loaded everytime when invoked.

### Testing

To see how the agent learned, type below:

```
python run.py
```

or

```
python run.py --render
```

This will iterate 10 episodes with trained model for testing.
Option '--render' will illustrate it with animation window at 30fps.

## Feature Highlights

Note: hyper-parameters below are not systematically determined.

- experience replay: capacity is 2048
- fixed target Q network: update interval is 3 epochs
- reward clipping: ranged by [0, 1] with sigmoid function
- fixed preprocess: all replay memory stores 4 frames each
- fully connected neural network with 1 hidden layer followed by relu non-linearity, optimized by Adam algorithm
	- minibatch size is 64
	- update interval is 10 frames as 1 epoch
	- input nodes are 12, hidden nodes are 32, output nodes are 2 that consist of leftmost & rightmost throttle as digital control
		- according to additional experiment, only 4 hidden nodes might be sufficient to solve this problem
- epsilon greedy: fixed to 5%, without decay
- action repeat: available, but disabled
- random agent & human agent (with usb gamepad) are also available

## Dependencies

- chainer==1.21.0
- gym==0.9.1
- numpy==1.12.0
- pygame==1.9.3

## Reference

- V. Mnih et al. Playing Atari with Deep Reinforcement Learning(2013).
- V. Mnih, K. Kavukcuoglu, D. Silver et al. Human-level control through deep reinforcement learning(2015).
