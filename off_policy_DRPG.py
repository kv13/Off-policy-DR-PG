import numpy as np 

class Tabular_MDP:
    def __init__(self, num_states, num_rows, num_cols, num_actions,  rewards, deterministic, transition_probs = [], gamma=0.9):
        self.num_states    = num_states
        self.num_rows      = num_rows
        self.num_cols      = num_cols
        self.num_actions   = num_actions
        self.deterministic = deterministic
        self.P             = transition_probs
        self.R             = rewards
        self.gamma         = gamma

    def step(self, state, action):
        
        if self.deterministic == True:
            if action == 0:
                next_state = state-1
                if state % self.num_cols == 0:  next_state = state
            elif action == 2:
                next_state = state + 1
                if state % self.num_cols == self.num_cols-1:
                    next_state = state 
            elif action == 1:
                next_state = state - self.num_cols
                if state < self.num_cols:
                    next_state = state
            elif action == 3:
                next_state = state + self.num_cols
                if state >= self.num_cols*(self.num_rows-1):
                    next_state = state
        else:
            next_state = np.random.choice(self.num_states, p=self.P[state, action])

        reward = self.R[state, action]
        return next_state, reward
    
class Policy:    
    def __init__(self, num_states, num_actions):
        self.num_states  = num_states
        self.num_actions = num_actions
        self.thetas      = 1/self.num_actions * np.ones((self.num_states, self.num_actions))

    def step(self, state):
        next_action =  np.random.choice(self.num_actions, p=self.thetas[state])
        return next_action

def policy_evaluation(policy, env, n=1000):
    delta = 0.999
    V = np.zeros(env.num_states)
    for s in range(env.num_states):
        for _ in range(n):
            a = policy.step(s)
            next_s, reward = env.step(s,a)
            V[s] += reward + delta * V[next_s]
    return V / n

def evaluate_G1(policy, wrt_theta, env, n=1000):
    
    V_hat = policy_evaluation(policy, env)
    delta = 0.999
    G1    = np.zeros(env.num_states)

    for s in range(env.num_states):
        for _ in range(n):
            a              = policy.step(s)
            next_s, reward = env.step(s, a) 
            G1[s]         += (wrt_theta == a) * (reward + delta * V_hat[next_s] - V_hat[s]) * 1/policy.thetas[wrt_theta]
    return G1/n

def evaluate_grad_Q(policy, wrt_theta, env, L=30, n_q=20):
    
    gamma_prime = 0.9
    delta       = 0.999
    grad_Q      = np.zeros(env.num_states, env.num_actions)

    V_hat       = policy_evaluation(policy, env)
    
    for s in range(env.num_states):
        for a in range(env.num_actions):
            for _ in range(n_q):
                state  = s
                action = a
                for k in range(L):
                    next_state, reward = env.step(state, action)
                    grad_Q[s,a]       += gamma_prime**k * (reward + delta * V_hat[next_state] - V_hat[state]) * (wrt_theta == action) * 1/policy.thetas[wrt_theta]
                    state              = next_state
                    action             = policy.step(state)
    
    return grad_Q/n_q


def evaluate_G2(policy, wrt_theta, env, n_u=20):

    G2      = np.zeros(env.num_states)
    Q       = evaluate_grad_Q(policy, wrt_theta, env)
    
    for s in range(env.num_states):
        for _ in range(n_u):
            action = policy.step(s)
            G2    += Q(s, action) 
    return G2 / n_u


def off_policy_DRPG(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode,delta = 0.999):
    
    for episode in range(num_episodes):
        state = np.random.choice(env.num_states) 
        
        # sample trajectory using bahavior policy
        rewards = []
        rhos    = []
        for t in range(max_steps_per_episode):
            action             = behavior_policy.step(state)
            next_state, reward = env.step(state, action)
            rewards.append(reward)
            rhos.append(behavior_policy.thetas[action] / target_policy.thetas[action])

        # calculate tilte values 
        
        V_hat  =
        Q_hat  =  
        
        for state in range(target_policy.num_states):
            for action in range(target_policy.num_actions):

                G1[state,action] = 
                G2               = 
                grad_Q           =  
            # update steps

        

            # rho = target_policy[state, action] / behavior_policy[state, action] 
            # V_hat = policy_evaluation(target_policy, env)  # Evaluate target policy
            # Q_hat = reward + delta * V_hat[next_state]     # Q-value estimate
            
            # # Update policy parameters using DR estimator
            # for a in range(env.num_actions):
            #     grad_log_pi = (a == action) - theta[state, a]
            #     theta[state, a] += 0.01 * rho * grad_log_pi * (reward + env.gamma * V_hat[next_state] - Q_hat)

            state = next_state
    return 0




    
