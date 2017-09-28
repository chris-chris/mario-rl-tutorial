#!/usr/bin/env python
import gym
import logging
import os
import sys
import gflags as flags

from baselines.acktr import acktr_disc
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import deepq

from baselines.acktr.policies import CnnPolicy

import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_boolean("prioritized", False, "prioritized_replay")
flags.DEFINE_boolean("dueling", False, "dueling")
flags.DEFINE_integer("num_cpu", 4, "number of cpus")

def train_acktr(env_id, num_timesteps, seed, num_cpu):
  """Train a acktr model.

    Parameters
    -------
    env_id: environment to train on
    num_timesteps: int
        number of env steps to optimizer for
    seed: int
        number of random seed
    num_cpu: int
        number of parallel agents

    """
  num_timesteps //= 4

  def make_env(rank):
    def _thunk():
      env = gym.make(env_id)
      env.seed(seed + rank)
      if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
      gym.logger.setLevel(logging.WARN)

      return MarioActionSpaceWrapper(env)
    return _thunk

  set_global_seeds(seed)
  env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

  policy_fn = CnnPolicy
  acktr_disc.learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu)
  env.close()

def train_dqn(env_id, num_timesteps):
  """Train a dqn model.

    Parameters
    -------
    env_id: environment to train on
    num_timesteps: int
        number of env steps to optimizer for

    """

  # 1. Create gym environment
  env = gym.make(FLAGS.env)
  # 2. Apply action space wrapper
  env = MarioActionSpaceWrapper(env)
  # 3. Apply observation space wrapper to reduce input size
  env = ProcessFrame84(env)
  # 4. Create a CNN model for Q-Function
  model = deepq.models.cnn_to_mlp(
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=FLAGS.dueling
  )
  # 5. Train the model
  act = deepq.learn(
    env,
    q_func=model,
    lr=1e-4,
    max_timesteps=FLAGS.timesteps,
    buffer_size=10000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    train_freq=4,
    learning_starts=10000,
    target_network_update_freq=1000,
    gamma=0.99,
    prioritized_replay=FLAGS.prioritized
  )
  act.save("mario_model.pkl")
  env.close()

def main():
  FLAGS(sys.argv)
  # Choose which RL algorithm to train.

  print("env : %s" % FLAGS.env)
  print("algorithm : %s" % FLAGS.algorithm)
  print("timesteps : %s" % FLAGS.timesteps)
  print("prioritized : %s" % FLAGS.prioritized)
  print("dueling : %s" % FLAGS.dueling)
  print("num_cpu : %s" % FLAGS.num_cpu)

  if(FLAGS.algorithm == "deepq"): # Use DQN
    train_dqn(env_id=FLAGS.env, num_timesteps=FLAGS.timesteps)

  elif(FLAGS.algorithm == "acktr"): # Use acktr
    train_acktr(FLAGS.env, num_timesteps=FLAGS.timesteps, seed=0, num_cpu=FLAGS.num_cpu)

if __name__ == '__main__':
  main()
