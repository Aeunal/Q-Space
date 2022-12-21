from baselines.a2c import a2c
import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

env = gym.make("Breakout-v4")





def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1]
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

from baselines.common.distributions import Pd, PdType, _matching_fc
class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)
    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=x)
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent_vector, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32

class MlpPolicy(object):
    def __init__(self, ob_space, ac_space, nbatch, nsteps, sess, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        self.action = ac_space
        actdim = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = CategoricalPdType(ac_space.n)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

from functools import partial 

sess = tf.compat.v1.Session()
ob_space = env.observation_space
ob_space = gym.spaces.Box(low=0, high=255, shape=(210*160*3,), dtype=np.int8)
ac_space = env.action_space
print(ob_space)
print(ac_space)
print(ob_space.sample().shape)
print(ac_space.n)

model_path = r"C:\Users\aeunal\Pictures\itü\ai\Project\models\third"

nbatch = 8
#policy = MlpPolicy(sess, ob_space=ob_space, ac_space=ac_space, nbatch=nbatch, nsteps=0)
env.num_envs = 1
#model= a2c.learn("mlp", env, seed=42, total_timesteps=5e4)
#model= a2c.learn("mlp", env, seed=38, total_timesteps=5e5)
model= a2c.learn("mlp", env, seed=42, total_timesteps=1e3, load_path=model_path)
#a2c.Model(partial(MlpPolicy, ob_space, ac_space), env, 1) 
#model.save(model_path)

obs = env.reset()
obs = obs.reshape((1,100800))
state = model.initial_state if hasattr(model, 'initial_state') else None
dones = np.zeros((1,))
episode_rew = np.zeros(1)
while True:
    if state is not None:
        actions, _, state, _ = model.step(obs,S=state, M=dones)
    else:
        actions, _, _, _ = model.step(obs)
    print(actions)
    #print(actions)
    #actions = [int(action/2.0) for action in actions]
    obs, rew, done, _ = env.step(actions)
    obs = obs.reshape((1,100800))
    episode_rew += rew
    env.render()
    done_any = done.any() if isinstance(done, np.ndarray) else done
    if done_any:
        for i in np.nonzero(done)[0]:
            print('episode_rew={}'.format(episode_rew[i]))
            episode_rew[i] = 0
        #break
        obs = env.reset()
        obs = obs.reshape((1,100800))
    
env.close()