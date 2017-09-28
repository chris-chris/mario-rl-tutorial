from baselines import deepq
from baselines.acktr.policies import CnnPolicy
import gym

import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

import gflags as flags
import sys

from pysc2.env import sc2_env

import baselines.common.tf_util as U

step_mul = 16
steps = 200

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")

def main():
  FLAGS(sys.argv)
  # 1. Create gym environment
  env = gym.make(FLAGS.env)
  # 2. Apply action space wrapper
  env = MarioActionSpaceWrapper(env)
  # 3. Apply observation space wrapper to reduce input size
  env = ProcessFrame84(env)
  act = deepq.load("mario_model.pkl")

  while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
      env.render()
      obs, rew, done, _ = env.step(act(obs[None])[0])
      episode_rew += rew
    print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()
