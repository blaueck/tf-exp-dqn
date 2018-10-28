import argparse
import importlib
import os

import numpy as np
import tensorflow as tf

from memory_pool import MemoryPool, PriorityMemoryPool
from rl_env import Env, StateRecorder
from custom_ops import (softmax, project_distribution,
                        state_preprocess, sample_action)
from utils import TimeMeter, to_bool


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


def memory_iter(pool, batch_size):
    while True:
        yield pool.sample(batch_size)


def main(args):

    # create env
    env = Env(args.game)

    with tf.name_scope('data'), tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()
        epsilon = tf.placeholder(dtype=tf.float32, name='epsilon')

        if not args.use_priority:
            memory_pool = MemoryPool(args.pool_size)
            output_dtypes = (tf.uint8, tf.int64, tf.float32, tf.bool, tf.uint8)
            output_shapes = ([None, 4, 84, 84], [None, ],
                             [None, ], [None, ], [None, 4, 84, 84])
        else:
            memory_pool = PriorityMemoryPool(args.pool_size)
            output_dtypes = (tf.uint8, tf.int64, tf.float32, tf.bool,
                             tf.uint8, tf.int64, tf.float32)
            output_shapes = ([None, 4, 84, 84], [None, ], [None, ], [None, ],
                             [None, 4, 84, 84], [None, ], [None, ])

        train_dataset = (tf.data.Dataset
                         .from_generator(
                             lambda: memory_iter(memory_pool, args.batch_size),
                             output_dtypes, output_shapes=output_shapes)
                         .prefetch(1))

        iterator = train_dataset.make_one_shot_iterator()

        if not args.use_priority:
            state, action, reward, done, next_state = iterator.get_next()
        else:
            state, action, reward, done, next_state, indices, priorities = iterator.get_next()

    with tf.name_scope('net'), tf.device('/device:GPU:0'):
        Net = importlib.import_module('models.{}'.format(args.model)).Net

        online_net = Net(env.n_action, name='online_net')
        target_net = Net(env.n_action, name='target_net')

        if args.model != 'c51_cnn':
            online_q = online_net(state_preprocess(state), True)
            target_q = target_net(state_preprocess(next_state), False)
        else:
            support = np.linspace(-args.vmax, args.vmax,
                                  args.n_atom, dtype='float32')

            online_logits = online_net(state_preprocess(state), True)
            online_q_distribution = softmax(online_logits, axis=2)
            online_q = tf.reduce_sum(online_q_distribution * support, axis=2)

            target_logits = target_net(state_preprocess(next_state), False)
            target_q_distribution = softmax(target_logits, axis=2)
            target_q = tf.reduce_sum(target_q_distribution * support, axis=2)

        # choice action
        max_action = tf.cond(
            tf.random_uniform(shape=[], dtype=tf.float32) < epsilon,
            lambda: sample_action(env.n_action),
            lambda: tf.argmax(online_q, axis=1)[0])

        sync_op = []
        for w_target, w_online in zip(target_net.global_variables, online_net.global_variables):
            sync_op.append(tf.assign(w_target, w_online, use_locking=True))
        sync_op = tf.group(*sync_op)

    with tf.name_scope('losses'), tf.device('/device:GPU:0'):
        if args.model != 'c51_cnn':
            # compute online net q value
            online_q_val = tf.reduce_sum(tf.one_hot(
                action, env.n_action, 1., 0.) * online_q, axis=1)

            # compute target value
            if args.double:
                online_action = tf.argmax(online_net(
                    state_preprocess(next_state), False), axis=1)
                Y = tf.reduce_sum(
                    tf.one_hot(online_action, env.n_action, 1., 0.) * target_q, axis=1)
            else:
                Y = tf.reduce_max(target_q, axis=1)

            target_q_max = reward + args.gamma * \
                Y * (1. - tf.cast(done, 'float32'))
            target_q_max = tf.stop_gradient(target_q_max)

            loss = tf.losses.huber_loss(
                labels=target_q_max, predictions=online_q_val,
                reduction=tf.losses.Reduction.NONE)
        else:
            batch_indices = tf.range(args.batch_size, dtype=tf.int64)
            indices = tf.stack([batch_indices, action], axis=1)

            online_q_val = tf.gather_nd(online_q, indices)
            online_a_logits = tf.gather_nd(online_logits, indices)

            projected_distribution = project_distribution(
                target_q_distribution, support, reward, done, args.gamma, args.batch_size)
            projected_distribution = tf.stop_gradient(projected_distribution)

            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=online_a_logits, labels=projected_distribution)

        if args.use_priority:
            print('using priority memory...')
            new_priority = tf.sqrt(loss + 1e-10)

            priorities_update_op = tf.py_func(
                memory_pool.set_priority, [indices, new_priority], [],
                name='priority_update_op')

            loss_weights = 1.0 / tf.sqrt(priorities + 1e-10)
            loss_weights = loss_weights / tf.reduce_max(loss_weights)
            loss = loss_weights * loss
        else:
            priorities_update_op = tf.no_op()

        loss = tf.reduce_mean(loss)

    with tf.name_scope('metrics') as scope:
        mean_q_val, mean_q_val_update_op = tf.metrics.mean(
            tf.reduce_mean(online_q_val))
        mean_loss, mean_loss_update_op = tf.metrics.mean(loss)

        episode_reward = tf.placeholder(tf.float32, shape=[])
        mean_episode_reward, mean_episode_reward_update_op = tf.metrics.mean(
            episode_reward)

        metrics_update_op = tf.group(mean_q_val_update_op, mean_loss_update_op)
        metrics_reset_op = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope))

        # collect metric summary alone, because it need to
        # summary after metrics update
        metric_summary = [
            tf.summary.scalar('loss', mean_loss, collections=[]),
            tf.summary.scalar('mean_q_val', mean_q_val, collections=[]),
            tf.summary.scalar('mean_episode_reward', mean_episode_reward, collections=[])]

    optimizer = tf.train.AdamOptimizer(args.learning_rate, epsilon=0.01/32)

    with tf.device('/device:GPU:0'):
        grad_and_v = optimizer.compute_gradients(
            loss, var_list=online_net.trainable_variables)

        train_op = optimizer.apply_gradients(
            grad_and_v, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(train_op, priorities_update_op, *update_ops)

    # build summary
    for g, v in grad_and_v:
        tf.summary.histogram(v.name + '/grad', g)
    for w in online_net.global_variables + target_net.global_variables:
        tf.summary.histogram(w.name, w)
    tf.summary.histogram('q_value', online_q)

    train_summary_str = tf.summary.merge_all()
    metric_summary_str = tf.summary.merge(metric_summary)

    # init op
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # prepare for the logdir
    if not tf.gfile.Exists(args.logdir):
        tf.gfile.MakeDirs(args.logdir)


    # summary writer
    train_writer = tf.summary.FileWriter(
        os.path.join(args.logdir, 'train'),
        tf.get_default_graph())

    # session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # do initialization
    sess.run(init_op)

    # saver
    saver = tf.train.Saver(max_to_keep=20)
    if args.restore:
        saver.restore(sess, args.restore)

    reward_sum = 0
    n_step = 0

    state_recorder = StateRecorder()
    tm = TimeMeter()

    start = sess.run(global_step) * args.update_period

    pre_ob = env.reset()

    sess.run([sync_op])
    for i in range(args.max_step):
        i_s = i + start

        if i_s < args.min_replay_history:
            eps = 1.
        else:
            eps = linearly_decaying_epsilon(
                args.epsilon_decay_period, i_s, args.min_replay_history, args.epsilon_train)

        state_recorder.add(pre_ob)

        a = sess.run(
            max_action,
            {state: state_recorder.state[None, ...], epsilon: eps})

        # step env
        ob, r, t, _ = env.step(a)
        n_step += 1
        reward_sum += r
        r = np.clip(r, -1, 1)

        # record pre observation, action, reward, False
        memory_pool.add(pre_ob, a, r, False)
        pre_ob = ob

        if args.render:
            env.render()

        if t:
            print('reward sum: {:.2f}, n step: {:d}'.format(
                reward_sum, n_step))
            sess.run([mean_episode_reward_update_op],
                     {episode_reward: reward_sum})

            # reset env
            memory_pool.add(pre_ob, 0, 0, True)

            pre_ob = env.reset()
            state_recorder.reset()
            reward_sum, n_step = 0., 0

        if i >= args.min_replay_history:
            if i_s % args.update_period == 0:
                tm.start()
                summary_fetch = (
                    [train_summary_str]
                    if abs((i_s % args.print_every - args.print_every) % args.print_every) < args.update_period
                    else [])
                ret = sess.run([train_op, metrics_update_op] + summary_fetch)
                tm.stop()

            if i_s % args.print_every == 0:

                ml, mq, mer = sess.run(
                    [mean_loss, mean_q_val, mean_episode_reward])

                print(('Step: {:d}, Mean Loss: {:.4f}, Mean Q: {:.4f}, '
                       'Mean Episode Reward: {:.2f}, Epsilon: {:.2f}, '
                       'Speed: {:.2f} i/s')
                      .format(i_s, ml, mq, mer, eps, args.batch_size / tm.get()))

                if len(ret) > 2:
                    train_writer.add_summary(ret[-1], sess.run(global_step))

                train_writer.add_summary(
                    sess.run(metric_summary_str), sess.run(global_step))

                tm.reset()

            if i_s % args.target_update_period == 0:
                sess.run([sync_op, metrics_reset_op])
                print('........sync........')

            if i_s % 10000 == 0:
                saver.save(sess, '{}/{}'.format(args.logdir, args.model),
                           global_step=sess.run(global_step), write_meta_graph=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dqn')
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--max_step', type=int, default=50000000)
    parser.add_argument('--game', default='Breakout-Atari2600')
    parser.add_argument('--pool_size', type=int, default=1000000)
    parser.add_argument('--min_replay_history', type=int, default=25000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--update_period', type=int, default=4)
    parser.add_argument('--target_update_period', type=int, default=10000)
    parser.add_argument('--epsilon_train', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_period', type=int, default=1000000)
    parser.add_argument('--model', default='c51_cnn')
    parser.add_argument('--logdir', default='log/c51_cnn')
    parser.add_argument('--restore', default='')
    parser.add_argument('--double', type=to_bool, default=True)
    parser.add_argument('--render', type=to_bool, default=True)
    parser.add_argument('--use_priority', type=to_bool, default=False)
    parser.add_argument('--n_atom', type=int, default=51)
    parser.add_argument('--vmax', type=float, default=10.)
    args = parser.parse_args()
    main(args)
