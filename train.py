#!/usr/bin/env python
import gym
import logging
import os
import sys
import gflags as flags

from baselines import bench
from baselines import logger
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from deepq import deepq
from deepq.models import cnn_to_mlp
from baselines.acktr.policies import CnnPolicy
from baselines.acktr import acktr_disc

import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pprint

FLAGS = flags.FLAGS
flags.DEFINE_string("log", "stdout", "logging type(stdout, tensorboard)")
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", False, "prioritized_replay")
flags.DEFINE_boolean("dueling", False, "dueling")
flags.DEFINE_integer("num_cpu", 4, "number of cpus")

max_mean_reward = 0

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
      # 1. Create gym environment
      env = gym.make(env_id)
      env.seed(seed + rank)
      if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
      gym.logger.setLevel(logging.WARN)
      # 2. Apply action space wrapper
      env = MarioActionSpaceWrapper(env)
      # 3. Apply observation space wrapper to reduce input size
      env = ProcessFrame84(env)

      return env
    return _thunk

  set_global_seeds(seed)
  env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

  policy_fn = CnnPolicy
  acktr_disc.learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu, save_interval=True)
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
  model = cnn_to_mlp(
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
    exploration_fraction=FLAGS.exploration_fraction,
    exploration_final_eps=0.01,
    train_freq=4,
    learning_starts=10000,
    target_network_update_freq=1000,
    gamma=0.99,
    prioritized_replay=FLAGS.prioritized,
    callback=deepq_callback
  )
  act.save("mario_model.pkl")
  env.close()

def deepq_callback(locals, globals):
  global max_mean_reward
  if('done' in locals and locals['done'] == True):
    if('mean_100ep_reward' in locals
       and locals['mean_100ep_reward'] > max_mean_reward
       ):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))
      if(not os.path.exists(os.path.join(PROJ_DIR,'models/deepq/'))):
        os.mkdir(os.path.join(PROJ_DIR,'models/'))
        os.mkdir(os.path.join(PROJ_DIR,'models/deepq/'))
      max_mean_reward = locals['mean_100ep_reward']
      act = deepq.ActWrapper(locals['act'], locals['act_params'])

      filename = os.path.join(PROJ_DIR,'models/deepq/mario_reward_%s.pkl' % locals['mean_100ep_reward'])
      act.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)

def main():
  FLAGS(sys.argv)
  logdir = "tensorboard"
  if(FLAGS.algorithm == "deepq"):
    logdir = "tensorboard/%s/%s_%s_%s_%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.exploration_fraction,
      FLAGS.prioritized,
      FLAGS.dueling,
    )
  elif(FLAGS.algorithm == "acktr"):
    logdir = "tensorboard/%s/%s_%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.num_cpu,
    )

  if(FLAGS.log == "tensorboard"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

  elif(FLAGS.log == "stdout"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[HumanOutputFormat(sys.stdout)])

  logger.info("env : %s" % FLAGS.env)
  logger.info("algorithm : %s" % FLAGS.algorithm)
  logger.info("timesteps : %s" % FLAGS.timesteps)
  logger.info("exploration_fraction : %s" % FLAGS.exploration_fraction)
  logger.info("prioritized : %s" % FLAGS.prioritized)
  logger.info("dueling : %s" % FLAGS.dueling)
  logger.info("num_cpu : %s" % FLAGS.num_cpu)

  # Choose which RL algorithm to train.
  if(FLAGS.algorithm == "deepq"): # Use DQN
    train_dqn(env_id=FLAGS.env, num_timesteps=FLAGS.timesteps)

  elif(FLAGS.algorithm == "acktr"): # Use acktr
    train_acktr(FLAGS.env, num_timesteps=int(FLAGS.timesteps), seed=0, num_cpu=FLAGS.num_cpu)

if __name__ == '__main__':
  main()
