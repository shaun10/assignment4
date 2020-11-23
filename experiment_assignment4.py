from time import time

import hiive.mdptoolbox.mdp as hmdp
import hiive.mdptoolbox.example as example
import collections
import pandas as pd
from joblib import dump

import gym
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11})

def frozen_lake_vi(env_name):
    env = gym.make(env_name)
    env = env.unwrapped
    desc = env.unwrapped.desc

    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10
    ### Value Iter ###
    print('Value Iter WITH FROZEN LAKE')
    best_vals = [0] * 10
    for i in range(0, 10):
        st = time()
        best_value, k = value_iteration(env, gamma=(i + 0.5) / 10)
        policy = extract_policy(env, best_value, gamma=(i + 0.5) / 10)
        policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000)
        gamma = (i + 0.5) / 10
        plot = plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Value Iter) ' + 'Gamma: ' + str(gamma),
            policy.reshape(4, 4), desc, colors_lake(), directions_lake()) if '4' in env_name else plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Value Iter) ' + 'Gamma: ' + str(gamma),
            policy.reshape(8, 8), desc, colors_lake(), directions_lake())
        end = time()
        gamma_arr[i] = (i + 0.5) / 10
        iters[i] = k
        best_vals[i] = best_value
        list_scores[i] = np.mean(policy_score)
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Frozen Lake - Value Iter - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Frozen Lake - Value Iter - Reward Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iters to Converge')
    plt.title('Frozen Lake - Value Iter - Convergence Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Convergence Analysis.png')
    plt.close()

    plt.plot(gamma_arr, best_vals)
    plt.xlabel('Gammas')
    plt.ylabel('Optimal Value')
    plt.title('Frozen Lake - Value Iter - Optimal Q Value Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Optimal Q Value Analysis.png')
    plt.close()



def frozen_lake_pi(env_name):
    env = gym.make(env_name)
    env = env.unwrapped
    desc = env.unwrapped.desc

    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10

    print('PI with frozen lake')
    
    for i in range(0, 10):
        st = time()
        best_policy, k = policy_iteration(env, gamma=(i + 0.5) / 10)
        scores = evaluate_policy(env, best_policy, gamma=(i + 0.5) / 10)

        plot = plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Policy Iteration) ' + 'Gamma: ' + str((i + 0.5) / 10),
            best_policy.reshape(4, 4), desc, colors_lake(), directions_lake()) if '4' in env_name else plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Policy Iteration) ' + 'Gamma: ' + str((i + 0.5) / 10),
            best_policy.reshape(8, 8), desc, colors_lake(), directions_lake())
        end = time()
        gamma_arr[i] = (i + 0.5) / 10
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Frozen Lake - Policy Iteration - Exec Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('Frozen Lake - Policy Iteration - Exec Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Frozen Lake - Policy Iter - Reward Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Policy Iter - Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iters to Converge')
    plt.title('Frozen Lake - Policy Iter - Convergence Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Policy Iter - Convergence Analysis.png')
    plt.close()




def frozen_lake_ql(env_name):
    env = gym.make(env_name)
    env = env.unwrapped
    desc = env.unwrapped.desc

    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10

    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')
    st = time()
    reward_array = []
    iter_array = []
    random_iter_array = []
    size_array = []
    chunks_array = []
    averages_array = []
    avg_iter_array = []
    avg_riter_array = []
    eps_array_out = []
    avg_eps = []
    time_array = []
    Q_array = []
    for epsl in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
        epsilon_current = epsl
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        random_iter_array = []
        eps_array_in = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        gamma = 0.99
        episodes = 30000
        for epis in range(episodes):
            state = env.reset()
            done = False
            t_reward = 0
            max_steps = 30000
            rand_iter = 0
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() > epsl:
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()
                    rand_iter += 1

                state, reward, done, info = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsl *= (2.71 ** (-epis / 10e10))
            rewards.append(t_reward)
            iters.append(i)
            random_iter_array.append(rand_iter)
            eps_array_in.append(epsl)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)
        eps_array_out.append(eps_array_in)

        env.close()
        end = time()
        time_array.append(end - st)

        # Plot results
        def chunked(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 10)
        chunks = list(chunked(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        chunks_array.append(chunks)
        averages_array.append(averages)

        chunks = list(chunked(iters, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        avg_iter_array.append(averages)

        chunks = list(chunked(random_iter_array, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        avg_riter_array.append(averages)

        chunks = list(chunked(eps_array_in, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        avg_eps.append(averages)

        plot = plot_policy_map(
            f'Frozen Lake Policy Map Q - Gamma {gamma} - Epsilon {epsilon_current}', np.argmax(Q, axis=1).reshape(4, 4),
            desc, colors_lake(), directions_lake())

    plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0], label='epsilon=0.05')
    plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1], label='epsilon=0.15')
    plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.25')
    plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='epsilon=0.50')
    plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4], label='epsilon=0.75')
    plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='epsilon=0.95')
    plt.legend()
    plt.xlabel('Iters')
    plt.grid()
    plt.title(f'Frozen Lake {env_name} - Q Learning - Rewards')
    plt.ylabel('Average Reward')
    plt.savefig(f'Frozen Lake {env_name} - Q Learning - Rewards.png')
    plt.close()

    plt.plot(range(0, len(reward_array[0]), size_array[0]), avg_eps[0], label='epsilon=0.05')
    plt.plot(range(0, len(reward_array[1]), size_array[1]), avg_eps[1], label='epsilon=0.15')
    plt.plot(range(0, len(reward_array[2]), size_array[2]), avg_eps[2], label='epsilon=0.25')
    plt.plot(range(0, len(reward_array[3]), size_array[3]), avg_eps[3], label='epsilon=0.50')
    plt.plot(range(0, len(reward_array[4]), size_array[4]), avg_eps[4], label='epsilon=0.75')
    plt.plot(range(0, len(reward_array[5]), size_array[5]), avg_eps[5], label='epsilon=0.95')
    plt.legend()
    plt.xlabel('Iters')
    plt.grid()
    plt.title(f'Frozen Lake {env_name} - Q Learning - Epsilon Decay')
    plt.ylabel('Epsilon Decay')
    plt.savefig(f'Frozen Lake {env_name} - Q Learning - Epsilon Decay.png')
    plt.close()

    for i, epsl in enumerate([0.05, 0.15, 0.25, 0.5, 0.75, 0.90]):
        plt.plot(range(0, len(reward_array[i]), size_array[i]), np.array([avg_iter_array[i], avg_riter_array[i]]).T)
        plt.ylim(0, 100)
        plt.legend(['Total Steps', 'Random Steps'], loc='best', title=f'epsilon={epsl}')
        plt.xlabel('Iters')
        plt.grid()
        plt.title(f'Frozen Lake - Q Learning steps - Epsilon = {epsl}')
        plt.ylabel('Step count')
        plt.savefig(f'Frozen Lake - Q Learning steps - Epsilon = {epsl}.png')
        plt.close()

    plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.95], time_array)
    plt.xlabel('Epsilon Values')
    plt.grid()
    plt.title('Frozen Lake - Q Learning')
    plt.ylabel('Execution Time (s)')
    plt.savefig(f'Frozen Lake {env_name} - Q Learning.png')
    plt.close()

    plt.subplot(1, 6, 1)
    plt.imshow(Q_array[0])
    plt.title('ε=0.05')

    plt.subplot(1, 6, 2)
    plt.title('ε=0.15')
    plt.imshow(Q_array[1])

    plt.subplot(1, 6, 3)
    plt.title('ε=0.25')
    plt.imshow(Q_array[2])

    plt.subplot(1, 6, 4)
    plt.title('ε=0.50')
    plt.imshow(Q_array[3])

    plt.subplot(1, 6, 5)
    plt.title('ε=0.75')
    plt.imshow(Q_array[4])

    plt.subplot(1, 6, 6)
    plt.title('ε=0.95')
    plt.imshow(Q_array[5])
    plt.colorbar()

    plt.savefig(f'Frozen Lake {env_name} - Q Learning - Varying Epsilon.png')
    plt.close()


def frozen_lake_exp(env_name):
    '''
    0 = left; 1 = down; 2 = right;  3 = up
    :param env_name:
    :return:
    '''

    env = gym.make(env_name)
    env = env.unwrapped
    desc = env.unwrapped.desc

    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10

    ### POLICY ITERATION ####
    print('PI frozen lake')
    for i in range(0, 10):
        st = time()
        best_policy, k = policy_iteration(env, gamma=(i + 0.5) / 10)
        scores = evaluate_policy(env, best_policy, gamma=(i + 0.5) / 10)
        plot = plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Policy Iteration) ' + 'Gamma: ' + str((i + 0.5) / 10),
            best_policy.reshape(8, 8), desc, colors_lake(), directions_lake()) if '8' in env_name else plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Policy Iteration) ' + 'Gamma: ' + str((i + 0.5) / 10),
            best_policy.reshape(4, 4), desc, colors_lake(), directions_lake())
        end = time()
        gamma_arr[i] = (i + 0.5) / 10
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Frozen Lake - Policy Iteration - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('Frozen Lake - Policy Iteration - Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Frozen Lake - Policy Iteration - Reward Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Policy Iteration - Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iters to Converge')
    plt.title('Frozen Lake - Policy Iteration - Convergence Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Policy Iteration - Convergence Analysis.png')
    plt.close()

    ### Value Iter ###
    print('Value Iter WITH FROZEN LAKE')
    best_vals = [0] * 10
    for i in range(0, 10):
        st = time()
        best_value, k = value_iteration(env, gamma=(i + 0.5) / 10)
        policy = extract_policy(env, best_value, gamma=(i + 0.5) / 10)
        policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000)
        gamma = (i + 0.5) / 10
        plot = plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Value Iter) ' + 'Gamma: ' + str(gamma),
            policy.reshape(8, 8), desc, colors_lake(), directions_lake()) if '8' in env_name else plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Value Iter) ' + 'Gamma: ' + str(gamma),
            policy.reshape(4, 4), desc, colors_lake(), directions_lake())
        end = time()
        gamma_arr[i] = (i + 0.5) / 10
        iters[i] = k
        best_vals[i] = best_value
        list_scores[i] = np.mean(policy_score)
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Frozen Lake - Value Iter - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Frozen Lake - Value Iter - Reward Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iters to Converge')
    plt.title('Frozen Lake - Value Iter - Convergence Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Convergence Analysis.png')
    plt.close()

    plt.plot(gamma_arr, best_vals)
    plt.xlabel('Gammas')
    plt.ylabel('Optimal Value')
    plt.title('Frozen Lake - Value Iter - Optimal Q Value Analysis')
    plt.grid()
    plt.savefig('Frozen Lake - Value Iter - Optimal Q Value Analysis.png')
    plt.close()

    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')
    st = time()
    reward_array = []
    iter_array = []
    size_array = []
    chunks_array = []
    averages_array = []

    time_array = []
    Q_array = []
    for epsil in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        gamma = 0.99
        episodes = 30000
        env = gym.make(env_name)
        env = env.unwrapped
        for episode in range(episodes):
            state = env.reset()
            done = False
            t_reward = 0
            max_steps = 1000000
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() < epsil:
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()

                state, reward, done, info = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsil = (1 - 2.71 ** (-episode / 1000))
            rewards.append(t_reward)
            iters.append(i)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        env.close()
        end = time()
        time_array.append(end - st)

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 50)
        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        chunks_array.append(chunks)
        averages_array.append(averages)

    plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0], label='epsil=0.05')
    plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1], label='epsil=0.15')
    plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsil=0.25')
    plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='epsil=0.50')
    plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4], label='epsil=0.75')
    plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='epsil=0.95')
    plt.legend()
    plt.xlabel('Iters')
    plt.grid()
    plt.title('Frozen Lake - Q Learning - Constant Epsilon')
    plt.ylabel('Average Reward')
    plt.savefig('Frozen Lake - Q Learning - Constant Epsilon.png')
    plt.close()

    plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.95], time_array)
    plt.xlabel('Epsilon Values')
    plt.grid()
    plt.title('Frozen Lake - Q Learning')
    plt.ylabel('Execution Time (s)')
    plt.savefig('Frozen Lake - Q Learning.png')
    plt.close()

    plt.subplot(1, 6, 1)
    plt.imshow(Q_array[0])
    plt.title('Epsilon=0.05')

    plt.subplot(1, 6, 2)
    plt.title('Epsilon=0.15')
    plt.imshow(Q_array[1])

    plt.subplot(1, 6, 3)
    plt.title('Epsilon=0.25')
    plt.imshow(Q_array[2])

    plt.subplot(1, 6, 4)
    plt.title('Epsilon=0.50')
    plt.imshow(Q_array[3])

    plt.subplot(1, 6, 5)
    plt.title('Epsilon=0.75')
    plt.imshow(Q_array[4])

    plt.subplot(1, 6, 6)
    plt.title('Epsilon=0.95')
    plt.imshow(Q_array[5])
    plt.colorbar()

    plt.savefig('Frozen Lake - Q Learning - Varying Epsilon.png')
    plt.close()


def run_episode(env, policy, gamma, render=True):
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=100):
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma):
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    v = np.zeros(env.nS)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if np.sum((np.fabs(prev_v - v))) <= eps:
            break
    return v


def policy_iteration(env, gamma):
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iters = 200000
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            k = i + 1
            break
        policy = new_policy
    return policy, k


def value_iteration(env, gamma):
    # initialize value-function
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    v = np.zeros(env.nS)
    max_iters = 100000
    eps = 1e-20
    desc = env.unwrapped.desc
    for i in range(max_iters):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        if np.sum(np.fabs(prev_v - v)) <= eps:
            k = i + 1
            break
    return v, k


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    # citation - https://github.com/tonyabracadabra/Frozen-Lake-RL/blob/master/rl.py
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.savefig(title + str('.png'))
    plt.close()

    return plt


def forest_pi():

    print('POLICY ITERATION WITH FOREST MANAGEMENT')
    P, R = example.forest(S=20)
    value_f = [0] * 10
    policy = [0] * 10
    iters = [0] * 10
    time_array = [0] * 10
    gamma_arr = [0] * 10
    for i in range(0, 10):
        pi = hmdp.PolicyIteration(P, R, (i + 0.5) / 10, max_iter=10000)
        pi.run()
        gamma_arr[i] = (i + 0.5) / 10
        value_f[i] = np.mean(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Forest Management - Policy Iteration - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('Forest Management - Policy Iteration - Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, value_f)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Forest Management - Policy Iteration - Reward Analysis')
    plt.grid()
    plt.savefig('Forest Management - Policy Iteration - Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iters to Converge')
    plt.title('Forest Management - Policy Iteration - Convergence Analysis')
    plt.grid()
    plt.savefig('Forest Management - Policy Iteration - Convergence Analysis.png')
    plt.close()


def forest_vi():
    print('Value Iter with Forest Management')
    P, R = example.forest(S=20)
    value_f = [0] * 10
    policy = [0] * 10
    iters = [0] * 10
    time_array = [0] * 10
    gamma_arr = [0] * 10

    for i in range(0, 10):
        pi = hmdp.ValueIteration(P, R, gamma=(i + 0.5) / 10)
        pi.run()
        gamma_arr[i] = (i + 0.5) / 10
        value_f[i] = np.mean(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Forest Management - Value Iter - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('Forest Management - Value Iter - Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, value_f)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Forest Management - Value Iter - Reward Analysis')
    plt.grid()
    plt.savefig('Forest Management - Value Iter - Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iters to Converge')
    plt.title('Forest Management - Value Iter - Convergence Analysis')
    plt.grid()
    plt.savefig('Forest Management - Value Iter - Convergence Analysis.png')
    plt.close()


def forest_ql(gamma):
    print('Q LEARNING WITH FOREST MANAGEMENT')
    P, R = example.forest(S=500, p=0.01)
    n_iters = int(1e6)
    stat_frequency = 1
    plot_range = int(n_iters / stat_frequency)
    value_f = []
    policy = []
    iters = []
    time_array = []
    Q_table = []
    rew_array = []
    error_array = []

    for epsil in [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 1]:
        print('Epsilon :', epsil, 'Gamma :', gamma)
        st = time()
        pi = hmdp.QLearning(P, R, gamma=gamma, epsilon=epsil, n_iter=n_iters, run_stat_frequency=stat_frequency)
        end = time()
        pi.run()
        rew_array.append(np.array(pi.v_mean).flatten())
        error_array.append(np.log(pi.error_mean))
        value_f.append(np.mean(pi.V))
        policy.append([epsil, gamma, pi.policy])
        time_array.append(end - st)
        Q_table.append(pi.Q)
        qlearner_stats = collections.defaultdict(list)
        for stats in pi.run_stats:
            qlearner_stats['state'].append(stats['State'])
            qlearner_stats['action'].append(stats['Action'])
            qlearner_stats['reward'].append(stats['Reward'])
            qlearner_stats['error'].append(stats['Error'])
            qlearner_stats['time'].append(stats['Time'])
            qlearner_stats['alpha'].append(stats['Alpha'])
            qlearner_stats['epsil'].append(stats['Epsilon'])
            qlearner_stats['max_v'].append(stats['Max V'])
            qlearner_stats['mean_v'].append(stats['Mean V'])
        qlearner_stats_df = pd.DataFrame(qlearner_stats)
        dump(qlearner_stats_df, f'{epsil}_{gamma}_qlearner_stats_forest.joblib')

    try:
        policy_df = pd.DataFrame(policy, columns=['eps', 'gamma', 'policy'])
        dump(policy_df, f'Q_forest_policy_{gamma}.joblib')
    except:
        pass

    plt.plot(range(0, plot_range), rew_array[0], label='epsil=0.05')
    plt.plot(range(0, plot_range), rew_array[1], label='epsil=0.15')
    plt.plot(range(0, plot_range), rew_array[2], label='epsil=0.25')
    plt.plot(range(0, plot_range), rew_array[3], label='epsil=0.50')
    plt.plot(range(0, plot_range), rew_array[4], label='epsil=0.75')
    plt.plot(range(0, plot_range), rew_array[5], label='epsil=0.95')
    plt.legend()
    plt.xlabel('Iters')
    plt.grid()
    plt.title('Forest Management - Q Learning Rewards- Decaying Epsilon')
    plt.ylabel('Average Reward')
    plt.savefig(f'Forest Management - Q Learning Rewards- Decaying Epsilon - {gamma}.png')
    plt.close()

    plt.plot(np.arange(100, plot_range + 1, 100), error_array[0], label='epsil=0.05')
    plt.plot(np.arange(100, plot_range + 1, 100), error_array[1], label='epsil=0.15')
    plt.plot(np.arange(100, plot_range + 1, 100), error_array[2], label='epsil=0.25')
    plt.plot(np.arange(100, plot_range + 1, 100), error_array[3], label='epsil=0.50')
    plt.plot(np.arange(100, plot_range + 1, 100), error_array[4], label='epsil=0.75')
    plt.plot(np.arange(100, plot_range + 1, 100), error_array[5], label='epsil=0.95')
    plt.legend()
    plt.xlabel('Iters')
    plt.grid()
    plt.title('Forest Management - Mean Q Val Update (Log) - Decaying Epsilon')
    plt.ylabel('Mean Q value updates (Log)')
    plt.savefig(f'Forest Management - Mean Q Val Updates- Decaying Epsilon - {gamma}.png')
    plt.close()



def directions_lake():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }

def colors_lake():
    return {
        b'S': 'black',
        b'F': 'skyblue',
        b'H': 'darkred',
        b'G': 'green',
    }

print('Experiments beginning ')
#Frozen lake 4x4 first
frozen_lake_exp('FrozenLake8x8-v0')
frozen_lake_vi('FrozenLake-v0')
frozen_lake_pi('FrozenLake-v0')
frozen_lake_ql('FrozenLake-v0')
#Frozen lake 8x8 second
frozen_lake_exp('FrozenLake8x8-v0')
frozen_lake_vi('FrozenLake8x8-v0')
frozen_lake_pi('FrozenLake8x8-v0')
frozen_lake_ql('FrozenLake8x8-v0')
#Second problem Forest
forest_pi()
forest_vi()
forest_ql(0.95)
forest_ql(0.99)
forest_ql(0.5)
forest_ql(0.3)
forest_ql(0.1)
print('Experimenting has concluded')
