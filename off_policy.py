import numpy as np 
import matplotlib.pyplot as plt

def compute_reward_to_go(rewards, rhos, t, gamma):
    total_reward = 0
    for i in range(t, len(rewards)):
        total_reward += gamma**(i-t) * np.prod(rhos[0:i+1]) * rewards[i]
    return total_reward

def policy_evaluation(policy, env, n=100):
    delta = 0.999
    V     = np.zeros(env.num_states)
    for i in range(1, n):
        for s in range(env.num_states):
            a              = policy.step(s)
            next_s, reward = env.step(s,a)

            V[s] = (i-1)*V[s] + reward + delta * V[next_s]
            V[s] = V[s]/i
    return V 

def Q_evaluation(policy, V_hat, env, delta = 0.999):
    Q = np.zeros((env.num_states,env.num_actions))
    for s in range(env.num_states):
        for a in range(env.num_actions):
            next_s, reward = env.step(s, a) 
            Q[s, a]        = reward + delta * V_hat[next_s]
    return Q

def evaluate_G1(policy, state, wrt_theta, V_hat, env, n=100):
    
    delta = 0.999
    G1    = np.zeros(env.num_states)

    for s in range(env.num_states):
        if s == state:
            for _ in range(n):
                a              = policy.step(s)
                next_s, reward = env.step(s, a) 
                G1[s]         += (reward + delta * V_hat[next_s] - V_hat[s]) * ((wrt_theta == a) - policy.probs[state, wrt_theta])
    return G1/n

def evaluate_grad_Q(policy, wrt_state, wrt_theta, V_hat, env, L=30, n_q=20):
    
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
                    if state == wrt_state:
                        grad_Q[s,a] += gamma_prime**k * (reward + delta * V_hat[next_state] - V_hat[state]) * ((wrt_theta == action) - policy.probs[wrt_state, wrt_theta])
                    state  = next_state
                    action = policy.step(state)
    return grad_Q/n_q


def evaluate_G2(policy, Q_grad, env, n_u=20):

    G2      = np.zeros(env.num_states)
    
    for s in range(env.num_states):
        for _ in range(n_u):
            action = policy.step(s)
            G2[s] += Q_grad[s, action]
    return G2 / n_u

def cum_reward(policy, env, T):
    
    num_seeds   = 5
    all_rewards = []
    
    for _ in range(num_seeds):
        state = np.random.choice(env.num_states) 

        for traj in range(5):
            total_reward_over_traj = 0
            for t in range(T):
                action                  = policy.step(state)
                next_state, reward      = env.step(state, action)
                total_reward_over_traj += reward
                state = next_state
            all_rewards.append(total_reward_over_traj)
    return np.mean(all_rewards), np.std(all_rewards)/np.sqrt(num_seeds)
            
def off_policy_DRPG(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, theta=0.9, delta = 0.999, lr=0.01):
    exp_rewards = []
    exp_std     = []
    
    for episode in range(num_episodes):
        
        state = np.random.choice(env.num_states) 
         
        print("episode:", episode)
        print('initial state:', state)
        print('probs target policy:', target_policy.probs)
        print('probs behavior policy: ', behavior_policy.probs)
        
        # sample trajectory using bahavior policy
        rewards = []
        rhos    = []
        states  = []
        actions = []
        for t in range(max_steps_per_episode):
            action = behavior_policy.step(state)
            states.append(state)
            actions.append(action)
            next_state, reward = env.step(state, action)
            rewards.append(reward)
            rhos.append(target_policy.probs[state, action] / behavior_policy.probs[state, action])
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
                    DRPG = (state == states[t]) * ((action == actions[t]) - target_policy.probs[state,action])*(sum1 - sum2) \
                         + _rho_prod*(G1[state, action, states[t]] + G2[state, action, states[t]]) \
                         - np.prod(rhos[0:t+1])*(grad_Q[state, action, states[t], actions[t]] - Q_hat[states[t], actions[t]] * (state == states[t]) * ((action == actions[t]) - target_policy.probs[state, action]))
                    target_policy.thetas[state,action] = target_policy.thetas[state,action] + lr*(env.gamma**t)*DRPG
            target_policy.probs[state,:] = np.exp(target_policy.thetas[state,:]) / sum(np.exp(target_policy.thetas[state,:]))
        
        cum, std = cum_reward(target_policy, env, max_steps_per_episode)
        cum_b, _ = cum_reward(behavior_policy, env,max_steps_per_episode)
        exp_rewards.append(cum) 
        exp_std.append(std)
        print("Total reward for behavior policy episode ",episode, 'is: ',cum_b)
        print("Total reward for target policy episode ",episode, 'is: ',cum)
    
    plt.plot(exp_rewards)
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards)-2*np.array(exp_std), np.array(exp_rewards)+2*np.array(exp_std), color='blue', alpha=0.2)
    plt.show()
    return 0

def off_policy_REINFORCE(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, lr=0.01):
    
    gamma       = 0.99
    exp_rewards = []
    exp_std     = []

    for episode in range(num_episodes): 

        state = np.random.choice(env.num_states)

        print("episode:", episode)
        print('initial state:', state)
        print('probs behaviour policy:', behavior_policy.probs)
        print('probs target policy:'   , target_policy.probs)

        # generate trajectory
        rewards = []
        states  = []
        actions = []
        rhos    = []
        for t in range(max_steps_per_episode):
            action = behavior_policy.step(state)
            states.append(state)
            actions.append(action)
            rhos.append(target_policy.probs[state, action] / behavior_policy.probs[state, action])

            next_state, reward = env.step(state, action)
            rewards.append(reward)
            state = next_state

        for t in range(max_steps_per_episode):
            reward_to_go = compute_reward_to_go(rewards, rhos, t, gamma)
            for s in range(target_policy.num_states):
                if s == states[t]:
                    for a in range(target_policy.num_actions):
                        # update thetas[state, action]     
                        param1 = (a == actions[t]) & (s == states[t])
                        target_policy.thetas[s, a] = target_policy.thetas[s, a] + lr * gamma**t * reward_to_go * (param1 - target_policy.probs[s, a])
        
        for s in range(target_policy.num_states):        
            target_policy.probs[s,:] = np.exp(target_policy.thetas[s,:]) / sum(np.exp(target_policy.thetas[s,:]))


        cum, std = cum_reward(target_policy, env, max_steps_per_episode)
        exp_rewards.append(cum)
        exp_std.append(std)
    plt.plot(exp_rewards)
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards)-1.96*np.array(exp_std), np.array(exp_rewards)+1.96*np.array(exp_std), color='gray', alpha=0.2)
    plt.show()

def off_policy_baseline(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, theta=0.9, delta=0.999, lr=0.01):
    exp_rewards = []
    exp_std     = []
    
    for episode in range(num_episodes):
        
        state = np.random.choice(env.num_states) 
         
        print("episode:", episode)
        print('initial state:', state)
        print('probs target policy:', target_policy.probs)
        print('probs behavior policy: ', behavior_policy.probs)
        
        # sample trajectory using bahavior policy
        rewards = []
        rhos    = []
        states  = []
        actions = []
        for t in range(max_steps_per_episode):
            action = behavior_policy.step(state)
            states.append(state)
            actions.append(action)
            next_state, reward = env.step(state, action)
            rewards.append(reward)
            rhos.append(target_policy.probs[state, action] / behavior_policy.probs[state, action])
            state = next_state

        # calculate tilte values 
        
        V_hat = policy_evaluation(target_policy, env)
        
        for state in range(target_policy.num_states):
            for action in range(target_policy.num_actions):
                PG = 0
                for t in range(max_steps_per_episode):
                    sum1 = 0
                    sum2 = 0
                    for t1 in range(t,max_steps_per_episode):
                        rho_prod_1 = np.prod(rhos[0:t1+1])
                        sum1 += rho_prod_1*(env.gamma*delta)**(t1-t)*rewards[t1]

                    for t2 in range(t+1,max_steps_per_episode):
                        rho_prod_2 = np.prod(rhos[0:t2])
                        sum2 += rho_prod_2*(theta*delta)**(t2-t)*(rhos[t2]-1)*V_hat[states[t2]]
                    
                    if t == 0: _rho_prod = 1
                    else: _rho_prod = np.prod(rhos[0:t])
                    PG = (state == states[t]) * ((action == actions[t]) - target_policy.probs[state,action])*(sum1 - sum2 \
                         - np.prod(rhos[0:t+1]) * env.gamma**t * V_hat[states[t]])
                        #  + _rho_prod*(G1[state, action, states[t]] + G2[state, action, states[t]]) \
                        #  - np.prod(rhos[0:t+1])*(grad_Q[state, action, states[t], actions[t]] - Q_hat[states[t], actions[t]] * (state == states[t]) * ((action == actions[t]) - target_policy.probs[state, action]))
                    target_policy.thetas[state,action] = target_policy.thetas[state,action] + lr*PG
            target_policy.probs[state,:] = np.exp(target_policy.thetas[state,:]) / sum(np.exp(target_policy.thetas[state,:]))
        
        cum, std = cum_reward(target_policy, env, max_steps_per_episode)
        cum_b, _ = cum_reward(behavior_policy, env,max_steps_per_episode)
        exp_rewards.append(cum) 
        exp_std.append(std)
        print("Total reward for behavior policy episode ",episode, 'is: ',cum_b)
        print("Total reward for target policy episode ",episode, 'is: ',cum)
    
    plt.plot(exp_rewards)
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards)-1.96*np.array(exp_std), np.array(exp_rewards)+1.96*np.array(exp_std), color='blue', alpha=0.2)
    plt.show()
    return 0

def off_policy_traj_cv(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, theta=0.9, delta=0.999, lr=0.01):
    exp_rewards = []
    exp_std     = []
    
    for episode in range(num_episodes):
        
        state = np.random.choice(env.num_states) 
         
        print("episode:", episode)
        print('initial state:', state)
        print('probs target policy:', target_policy.probs)
        print('probs behavior policy: ', behavior_policy.probs)
        
        # sample trajectory using bahavior policy
        rewards = []
        rhos    = []
        states  = []
        actions = []
        for t in range(max_steps_per_episode):
            action = behavior_policy.step(state)
            states.append(state)
            actions.append(action)
            next_state, reward = env.step(state, action)
            rewards.append(reward)
            rhos.append(target_policy.probs[state, action] / behavior_policy.probs[state, action])
            state = next_state

        # calculate tilte values 
        
        V_hat = policy_evaluation(target_policy, env)
        Q_hat = Q_evaluation(target_policy, V_hat, env) 
        
        G1     = np.zeros((target_policy.num_states, target_policy.num_actions, target_policy.num_states))
        # G2     = np.zeros((target_policy.num_states, target_policy.num_actions, target_policy.num_states))
        # grad_Q = np.zeros((target_policy.num_states, target_policy.num_actions, target_policy.num_states,  target_policy.num_actions))
        
        for state in range(target_policy.num_states):
            for action in range(target_policy.num_actions):

                G1[state, action, :]        = evaluate_G1(target_policy, state, action, V_hat, env)
                # grad_Q[state, action, :, :] = evaluate_grad_Q(target_policy, state, action, V_hat, env)
                # G2[state, action, :]        = evaluate_G2(target_policy, grad_Q[state, action], env)
        
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
                    DRPG = (state == states[t]) * ((action == actions[t]) - target_policy.probs[state,action])*(sum1 - sum2) \
                         + _rho_prod*(G1[state, action, states[t]]) \
                         - np.prod(rhos[0:t+1])*(Q_hat[states[t], actions[t]] * (state == states[t]) * ((action == actions[t]) - target_policy.probs[state, action]))
                    target_policy.thetas[state,action] = target_policy.thetas[state,action] + lr*(env.gamma**t)*DRPG
            target_policy.probs[state,:] = np.exp(target_policy.thetas[state,:]) / sum(np.exp(target_policy.thetas[state,:]))
        
        cum, std = cum_reward(target_policy, env, max_steps_per_episode)
        cum_b, _ = cum_reward(behavior_policy, env,max_steps_per_episode)
        exp_rewards.append(cum) 
        exp_std.append(std)
        print("Total reward for behavior policy episode ",episode, 'is: ',cum_b)
        print("Total reward for target policy episode ",episode, 'is: ',cum)
    
    plt.plot(exp_rewards)
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards)-1.96*np.array(exp_std), np.array(exp_rewards)+1.96*np.array(exp_std), color='blue', alpha=0.2)
    plt.show()
    return 0