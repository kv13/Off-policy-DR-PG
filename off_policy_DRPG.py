import numpy as np 

class Tabular_MDP:
    def __init__(self, num_states, num_rows, num_cols, num_actions,  rewards, deterministic, transition_probs = [], gamma=1):
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
        self.thetas      = np.ones((self.num_states, self.num_actions))
        self.probs       = np.zeros((self.num_states, self.num_actions))
        for state in range(self.num_states):
            self.probs[state, :] = np.exp(self.thetas[state,:]) / sum(np.exp(self.thetas[state,:]))  

    def step(self, state):
        next_action =  np.random.choice(self.num_actions, p=self.probs[state])
        return next_action

def policy_evaluation(policy, env, n=100):
    delta = 0.999
    V = np.zeros(env.num_states)
    for i in range(1, n):
        for s in range(env.num_states):
            a = policy.step(s)
            next_s, reward = env.step(s,a)

            V[s] = (i-1)*V[s] + reward + delta * V[next_s]
            V[s] = V[s]/i
    return V 

def Q_evaluation(policy, V_hat, env, delta = 0.999):
    Q = np.zeros((env.num_states,env.num_actions))
    for s in range(env.num_states):
        for a in range(env.num_actions):
            next_s, reward = env.step(s, a) 
            Q[s, a]        = reward +  delta * V_hat[next_s]
    return Q

def evaluate_G1(policy, state, wrt_theta, V_hat, env, n=100):
    
    delta = 0.999
    G1    = np.zeros(env.num_states)

    for s in range(env.num_states):
        for _ in range(n):
            a              = policy.step(s)
            next_s, reward = env.step(s, a) 
            G1[s]         += (reward + delta * V_hat[next_s] - V_hat[s]) * ((wrt_theta == a) - policy.probs[state, wrt_theta])
    return G1/n

def evaluate_grad_Q(policy, state, wrt_theta, V_hat, env, L=30, n_q=20):
    
    gamma_prime = 0.9
    delta       = 0.999
    grad_Q      = np.zeros((env.num_states, env.num_actions))
    
    for s in range(env.num_states):
        for a in range(env.num_actions):
            for _ in range(n_q):
                state  = s
                action = a
                for k in range(L):
                    next_state, reward = env.step(state, action)
                    grad_Q[s,a]       += gamma_prime**k * (reward + delta * V_hat[next_state] - V_hat[state]) * ((wrt_theta == action) - policy.probs[state,wrt_theta])
                    state              = next_state
                    action             = policy.step(state)
    
    return grad_Q/n_q


def evaluate_G2(policy, Q_grad, env, n_u=20):

    G2      = np.zeros(env.num_states)
    
    for s in range(env.num_states):
        for _ in range(n_u):
            action = policy.step(s)
            G2[s]    += Q_grad[s, action]
    return G2 / n_u

def cum_reward(policy, env):
    V = policy_evaluation(policy, env)
    total_rew = 0
    for s in range(env.num_states):
        total_rew += V[s]
    return total_rew/env.num_states

def off_policy_DRPG(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, theta=0.9, delta = 0.999, lr=0.01):
    
    for episode in range(num_episodes):
        # state = np.random.choice(env.num_states) 
        state = env.num_cols * int(env.num_rows/2) + int(env.num_cols/2) 
        print('initial state', state)
        print('thetas ', target_policy.probs)
        
        # sample trajectory using bahavior policy
        rewards = []
        rhos    = []
        states  = []
        actions = []
        for t in range(max_steps_per_episode):
            action             = behavior_policy.step(state)
            states.append(state)
            actions.append(action)
            next_state, reward = env.step(state, action)
            rewards.append(reward)
            rhos.append(behavior_policy.probs[state, action] / (target_policy.probs[state, action] + 1e-6))
            state = next_state

        # calculate tilte values 
        
        V_hat = policy_evaluation(target_policy, env)
        Q_hat = Q_evaluation(target_policy, V_hat, env) 
        
        G1     = np.zeros((target_policy.num_states, target_policy.num_actions, target_policy.num_states))
        G2     = np.zeros((target_policy.num_states, target_policy.num_actions, target_policy.num_states))
        grad_Q = np.zeros((target_policy.num_states, target_policy.num_actions, target_policy.num_states,  target_policy.num_actions))
        
        for state in range(target_policy.num_states):
            for action in range(target_policy.num_actions):

                G1[state, action, :]        = evaluate_G1(target_policy, state, action, V_hat, env)
                grad_Q[state, action, :, :] = evaluate_grad_Q(target_policy, state, action, V_hat, env)
                G2[state, action, :]        = evaluate_G2(target_policy, grad_Q[state, action], env)

        # print(G1)
        # 
        # DRPG = 0
        # for t in range(max_steps_per_episode):
        
        for state in range(target_policy.num_states):
            for action in range(target_policy.num_actions):
                DRPG = 0
                for t in range(max_steps_per_episode):
                    sum1 = 0
                    sum2 = 0
                    for t1 in range(t,max_steps_per_episode):
                        rho_prod_1 = np.prod(rhos[0:t1+1])
                        sum1 += rho_prod_1*(env.gamma*delta)**(t1-t)*rewards[t1]

                    for t2 in range(t+1,max_steps_per_episode):
                        rho_prod_2 = np.prod(rhos[0:t2])
                        sum2 += rho_prod_2*(theta*delta)**(t2-t)*(rhos[t2]*Q_hat[states[t2], actions[t2]] - V_hat[states[t2]])
                    
                    if t == 0: _rho_prod = 1
                    else: _rho_prod = np.prod(rhos[0:t])
                    DRPG += ((action == actions[t]) - target_policy.probs[state,action])*(sum1 - sum2) \
                         + _rho_prod*(G1[state, action, states[t]] + G2[state, action, states[t]]) \
                         - np.prod(rhos[0:t+1])*(grad_Q[state, action, states[t], actions[t]] - Q_hat[states[t], actions[t]]*((action == actions[t]) - target_policy.probs[state,action]))
                print(DRPG)
                target_policy.thetas[state,action] = target_policy.thetas[state,action] + lr*DRPG
            
            # print(target_policy.thetas[state,:])
            target_policy.probs[state,:] = np.exp(target_policy.thetas[state,:]) / sum(np.exp(target_policy.thetas[state,:]))
        print("Total reward for behavior policy episode ",episode, 'is: ',cum_reward(behavior_policy, env))
        print("Total reward for target policy episode ",episode, 'is: ',cum_reward(target_policy, env))
    return 0


    
