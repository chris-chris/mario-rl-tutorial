# mario-rl-tutorial
#### **Gym Super Mario is an environment bundle for OpenAI Gym**
---
<div id="installation"></div>Installation
============

```shell
pip3 install git+https://github.com/chris-chris/gym-super-mario
pip3 install git+https://github.com/openai/baselines
```

 To load and run the environments, run

```python
import gym
import ppaquette_gym_super_mario
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
```

To train the model, run

```shell

python3 train.py

```
