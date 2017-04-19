#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, default="experts/Hopper-v1.pkl")
    parser.add_argument('--envname', type=str, default="Hopper-v1")
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=1000,
                        help='Number of expert roll outs')
    parser.add_argument('--out_file', type=str, help='save expert data to file')
    args = vars(parser.parse_args())

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args['expert_policy_file'])
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args['envname'])
        max_steps = args['max_timesteps'] or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args['num_rollouts']):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args['render']:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        if args['out_file'] is not None:
            pickle.dump(expert_data, open(args['out_file'], 'wb'))

if __name__ == '__main__':
    main()
