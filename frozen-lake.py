import numpy as np
import gym
import time

np.set_printoptions(suppress=True)

decay = 0.9

def get_policy_from_values(env, V):
    policy = np.zeros(env.nS, dtype=int)

    for s in range(env.nS):
        utilities = []
        for a in range(env.nA):
            action_value = 0

            for i in range(len(env.P[s][a])):
                prob, next_state, reward, done = env.P[s][a][i]

                action_value += prob * reward + (decay * prob * V[next_state])

            utilities.append(action_value)
        
        policy[s] = int(np.argmax(utilities))
    
    return policy

def get_values_from_policy(env, policy):
    A = np.eye(env.nS)
    b = np.zeros(env.nS)

    for s in range(env.nS):
        action = policy[s]

        for i in range(len(env.P[s][action])):
            prob, next_state, reward, _ = env.P[s][action][i]

            A[s][next_state] -= decay * prob

            b[s] += prob * reward
    
    return np.linalg.solve(A, b)

def value_iteration(env):
    V = np.random.random(env.nS)

    done = False

    for iter in range(10000):
        done = True
        for s in range(env.nS):
            q_values = []
            for a in range(env.nA):
                state_value = 0

                for i in range(len(env.P[s][a])):
                    prob, next_state, reward, done = env.P[s][a][i]

                    state_value += prob * reward + (decay * prob * V[next_state])
                
                q_values.append(state_value)
            
            best = max(q_values)

            if (abs(V[s]-best)>0.0001):
                done = False

            V[s] = best

        if (done):
            print(iter, " iterations required to converge")
            break

    policy = get_policy_from_values(env, V)
    
    return V, policy

def policy_iteration(env):
    policy = np.random.randint(env.nA, size = env.nS)

    for i in range(1000):
        V = get_values_from_policy(env, policy)
        new_policy = get_policy_from_values(env, V)

        done = True

        for s in range(len(new_policy)):
            if (new_policy[s]!=policy[s]):
                policy[s] = new_policy[s]
                done = False

        if (done):
            print(i, " iterations required to converge")
            break


    V = get_values_from_policy(env, policy)

    return V, policy

def q_learning(env):
    q_values = np.full((env.nS, env.nA), 0.5)
    
    epsilon = 0.7
    min_epsilon = 0.15
    epsilon_decay = 0.001
    visits = {}

    for i in range(20000):
        env.reset()
        env.seed(time.time_ns())
        done = False
        state = 0

        while not done:
            action = env.action_space.sample()

            if (np.random.random() > epsilon):
                action = np.argmax(q_values[state])

            if ((state, action) not in visits):
                visits[(state, action)] = 0
            
            learn_rate = 1/(1+visits[(state, action)])
            visits[(state, action)] += 1
            learn_rate = 0.05
            
            new_state, reward, done, _ = env.step(action)

            q_values[state, action] = (1-learn_rate)*q_values[state, action] + learn_rate*(reward + decay * max(q_values[new_state]))

            state = new_state

            if (done):
                q_values[state] = [0,0,0,0]
    
        epsilon = max(min_epsilon, 0.7*np.exp(-epsilon_decay*i))

    return q_values            

def simulate_policy(env, policy, render=False):
    env.reset()
    env.seed(time.time_ns())
    
    done = False

    state = 0

    reward = 0

    steps = 0

    while not done:
        state, reward, done, _ = env.step(policy[state])
        steps += 1

        if (render):
            env.render()

    return reward, steps

def evaluate_policy(env, policy):
    total_reward = 0
    total_steps = 0

    n = 1000

    for i in range(n):
        reward, steps = simulate_policy(env, policy)
        total_reward += reward
        total_steps += steps

    avg_reward = total_reward/n
    avg_steps = total_steps/n

    return avg_reward, avg_steps

def simulate_policy_Q(env, q_table, render=False):
    env.reset()
    env.seed(time.time_ns())
    
    done = False

    state = 0

    reward = 0

    steps = 0

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        steps += 1

        if (render):
            env.render()

    return reward, steps

def evaluate_policy_Q(env, q_table):
    total_reward = 0
    total_steps = 0

    n = 1000

    for i in range(n):
        reward, steps = simulate_policy_Q(env, q_table)
        total_reward += reward
        total_steps += steps

    avg_reward = total_reward/n
    avg_steps = total_steps/n

    return avg_reward, avg_steps

if (__name__=='__main__'):
    env = gym.make("FrozenLake-v1")
    env.seed(0)
    env.reset()
    
    start = time.time()
    V, policy = value_iteration(env)
    total = time.time() - start
    print(total)
    print(evaluate_policy(env, policy))

    start = time.time()
    V, policy = policy_iteration(env)
    total = time.time() - start
    print(total)
    print(evaluate_policy(env, policy))

    q_table = q_learning(env)
    print(evaluate_policy_Q(env, q_table))

    env.reset()
    env.render()
