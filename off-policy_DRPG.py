import numpy as np 

class Tabular_MDP:

    def __init__(self, num_states, num_actions, transition_probs, rewards, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = transition_probs
        self.R = rewards
        self.gamma = gamma

    def step(self, state, action):
        next_state = np.random.choice(self.num_states, p=self.P[state, action])
        reward = self.R[state, action]
        return next_state, reward
    

def policy_evaluation(policy, env, n=1000):
    delta = 0.999
    V = np.zeros(env.num_states)
    for _ in range(n):
        for s in range(env.num_states):
            # for a in range(env.num_actions):
                
                next_s, reward = env.step(s, a)
                V[s] += policy[s, a] * (reward + delta * V[next_s])
    return V / n

def evaluate_G1(policy, env, n=1000):
    V_hat = policy_evaluation(policy, env)
    delta = 0.999
    G1 = np.zeros(env.num_states)
    for _ in range(n):
        for s in range(env.num_states):
            for a in range(env.num_actions):
                next_s, reward = env.step(s, a)
                G1[s] += policy[s, a] * (reward + delta*V_hat[next_s]-V_hat[s])
    return G1/n

def evaluate_grad_Q(policy, env, L=30, n_q=20):
    gamma_prime = 0.9
    grad_Q = np.zeros(env.num_states, env.num_actions)



def off_policy_DRPG(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode):
    delta = 0.999
    theta = np.random.rand(env.num_states, env.num_actions)
    theta /= theta.sum(axis=1, keepdims=True)

    for episode in range(num_episodes):
        state = np.random.choice(env.num_states) 
        for t in range(max_steps_per_episode):
            action = np.random.choice(env.num_actions, p=behavior_policy[state])
            next_state, reward = env.step(state, action)
            
            rho = target_policy[state, action] / behavior_policy[state, action] 
            V_hat = policy_evaluation(target_policy, env)  # Evaluate target policy
            Q_hat = reward + delta * V_hat[next_state]  # Q-value estimate
            
            # Update policy parameters using DR estimator
            for a in range(env.num_actions):
                grad_log_pi = (a == action) - theta[state, a]
                theta[state, a] += 0.01 * rho * grad_log_pi * (reward + env.gamma * V_hat[next_state] - Q_hat)

            state = next_state
    return theta




    
