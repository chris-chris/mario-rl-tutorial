from deepq import deepq
from acktr.policies import CnnPolicy
from acktr import acktr_disc
import gym

import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

import gflags as flags
import sys

import numpy as np

from pysc2.env import sc2_env

import baselines.common.tf_util as U

step_mul = 16
steps = 200

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_string("file", "mario_reward_930.6.pkl", "Trained model file to use.")

def main():
  FLAGS(sys.argv)
  # 1. Create gym environment
  env = gym.make(FLAGS.env)
  # 2. Apply action space wrapper
  env = MarioActionSpaceWrapper(env)
  # 3. Apply observation space wrapper to reduce input size
  env = ProcessFrame84(env)

  if(FLAGS.algorithm == "deepq"):

    act = deepq.load("models/deepq/%s" % FLAGS.file)
    nstack = 4
    nh, nw, nc = env.observation_space.shape
    history = np.zeros((1, nh, nw, nc*nstack), dtype=np.uint8)

    while True:
      obs, done = env.reset(), False
      history = update_history(history, obs)
      episode_rew = 0
      while not done:
        env.render()
        action = act(history)[0]
        obs, rew, done, _ = env.step(action)
        history = update_history(history, obs)
        episode_rew += rew
        print("action : %s reward : %s" % (action, rew))

      print("Episode reward", episode_rew)

  elif(FLAGS.algorithm == "acktr"):

    policy_fn = CnnPolicy
    model = acktr_disc.load(policy_fn, env, seed=0, total_timesteps=1,
                     nprocs=4, filename="models/acktr/%s" % FLAGS.file)
    nstack = 4
    nh, nw, nc = env.observation_space.shape
    history = np.zeros((1, nh, nw, nc*nstack), dtype=np.uint8)

    while True:
      obs, done = env.reset(), False
      history = update_history(history, obs)
      episode_rew = 0
      while not done:
        env.render()
        action = model.step(history)[0][0]
        obs, rew, done, _ = env.step(action)
        history = update_history(history, obs)
        episode_rew += rew
        print("action : %s reward : %s" % (action, rew))
      print("Episode reward", episode_rew)

def update_history(history, obs):
  obs = np.reshape(obs, (1,84,84,1))
  history = np.roll(history, shift=-1, axis=3)
  history[:, :, :, -1] = obs[:, :, :, 0]
  return history

if __name__ == '__main__':
  main()
