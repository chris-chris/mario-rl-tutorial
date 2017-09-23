import numpy as np
import tensorflow as tf
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class CnnPolicy(object):

  def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
    nbatch = nenv*nsteps
    nh, nw, nc = ob_space.shape
    ob_shape = (nbatch, nh, nw, nc*nstack)
    nact = 14
    X = tf.placeholder(tf.uint8, ob_shape) #obs
    with tf.variable_scope("model", reuse=reuse):
      h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
      h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
      h3 = conv(h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
      h3 = conv_to_fc(h3)
      h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
      pi = fc(h4, 'pi', nact, act=lambda x:x)
      vf = fc(h4, 'v', 1, act=lambda x:x)

    v0 = vf[:, 0]
    a0 = sample(pi)
    self.initial_state = [] #not stateful

    def step(ob, *_args, **_kwargs):
      a, v = sess.run([a0, v0], {X:ob})
      return a, v, [] #dummy state

    def value(ob, *_args, **_kwargs):
      return sess.run(v0, {X:ob})

    self.X = X
    self.pi = pi
    self.vf = vf
    self.step = step
    self.value = value

