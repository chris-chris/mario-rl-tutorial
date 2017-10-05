# mario-rl-tutorial
#### **Gym Super Mario is an environment bundle for OpenAI Gym**
---

## Installation

```shell
pip3 install git+https://github.com/chris-chris/gym-super-mario
pip3 install git+https://github.com/openai/baselines
```

### To load and run the environments, run

```python
import gym
import ppaquette_gym_super_mario
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
```

### To train the model, run

```shell

python3 train.py

```

## Paramters

You can also customize the training with parameters.

```shell

python3 train.py --algorithm=deepq --timesteps=2000000 --log=stdout --env=ppaquette/SuperMarioBros-1-1-v0

```

|                      | Description                                     | Default                         | Parameter Type |
|----------------------|-------------------------------------------------|---------------------------------|----------------|
| env                  | Gym Environment                                 | ppaquette/SuperMarioBros-1-1-v0 | string         |
| log                  | logging type  : tensorboard, stdout             | stdout                          | string         |
| algorithm            | Currently, support 2 algorithms  : deepq, acktr | deepq                           | string         |
| timesteps            | Total training steps                            | 2000000                         | int            |
| exploration_fraction | exploration fraction                            | 0.5                             | float          |
| prioritized          | Whether using prioritized replay for DQN        | False                           | boolean        |
| dueling              | Whether using dueling network for DQN           | False                           | boolean        |
| num_cpu              | number of agents for A3C(acktr)                 | 4                               | int            |
