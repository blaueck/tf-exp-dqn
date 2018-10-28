import argparse
import importlib
import os

import numpy as np
import tensorflow as tf

from utils import Env, StateRecorder
from custom_ops import softmax, state_preprocess, sample_action


def main(args):

    # create env
    if args.record_dir:
        record_dir = args.record_dir
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        elif not os.path.isdir(record_dir):
            raise Exception("{:s} is not a directory.".format(record_dir))
    else:
        record_dir = False
    
    env = Env(args.game, record=record_dir)
    
    state = tf.placeholder(tf.uint8, shape=[None, 4, 84, 84])

    Net = importlib.import_module('models.{}'.format(args.model)).Net
    online_net = Net(env.n_action, name='online_net')

    if args.model != 'c51_cnn':
        online_q = online_net(state_preprocess(state), False)
    else:
        support = np.linspace(-args.vmax, args.vmax, args.n_atom, dtype='float32')

        online_logits = online_net(state_preprocess(state), False)
        online_q_distribution = softmax(online_logits, axis=2)
        online_q = tf.reduce_sum(online_q_distribution * support, axis=2)

    # choice action
    max_action = tf.cond(
        tf.random_uniform(shape=[], dtype=tf.float32) < args.epsilon,
        lambda: sample_action(env.n_action),
        lambda: tf.argmax(online_q, axis=1)[0])
    
    # saver
    saver = tf.train.Saver(max_to_keep=20)

    # session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    saver.restore(sess, args.checkpoint)

    recorder = StateRecorder()
    recorder.add(env.reset())

    total_reward = 0.
    for i in range(args.max_step):
        a = sess.run(max_action, {state: recorder.state[None, ...]})

        ob, reward, done, _ = env.step(a)
        recorder.add(ob)
        total_reward += reward

        if args.render:
            env.render()

        if done:
            break

    print('Total Reward: {:.2f}, N Step: {:d}'.format(total_reward, i + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dqn')
    parser.add_argument('--game', default='Breakout-Atari2600')
    parser.add_argument('--model', default='c51_cnn')
    parser.add_argument('--checkpoint', default='./log/c51_cnn/c51_cnn-1702501')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--n_atom', type=int, default=51)
    parser.add_argument('--vmax', type=float, default=10.)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--max_step', type=int, default=10000)
    parser.add_argument('--record_dir', default='')
    args = parser.parse_args()
    main(args)
