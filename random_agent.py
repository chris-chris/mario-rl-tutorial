import gflags as flags
import sys
import gym

import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper, ProcessFrame84

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")

class RandomAgent(object):
  """The world's simplest agent!"""
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward, done):
    return self.action_space.sample()

def main():
  FLAGS(sys.argv)
  # Choose which RL algorithm to train.

  print("env : %s" % FLAGS.env)

  # 1. Create gym environment
  env = gym.make(FLAGS.env)
  # 2. Apply action space wrapper
  env = MarioActionSpaceWrapper(env)
  # 3. Apply observation space wrapper to reduce input size
  env = ProcessFrame84(env)

  agent = RandomAgent(env.action_space)

  episode_count = 100
  reward = 0
  done = False

  for i in range(episode_count):
    ob = env.reset()
    while True:
      action = agent.act(ob, reward, done)
      ob, reward, done, _ = env.step(action)
      if done:
        break

if __name__ == '__main__':
  main()
