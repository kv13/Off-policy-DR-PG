import numpy as np
import matplotlib.pyplot as plt
from off_policy import compute_reward_to_go, policy_evaluation, Q_evaluation, evaluate_G1, evaluate_grad_Q, evaluate_G2, cum_reward

def off_policy_DRPG(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, rewards, states, rhos, actions, 
                    theta=0.9, delta = 0.999, lr=0.01):
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
                    target_policy.thetas[state,action] = target_policy.thetas[state,action] + lr*env.gamma**t*DRPG
            target_policy.probs[state,:] = np.exp(target_policy.thetas[state,:]) / sum(np.exp(target_policy.thetas[state,:]))
        
        #exp_rewards.append(cum_reward(target_policy, env, max_steps_per_episode)) 
        cum, std = cum_reward(target_policy, env, max_steps_per_episode)
        return cum, std


def off_policy_REINFORCE(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, 
                         rewards, states, rhos, actions, lr=0.01):
        
        gamma       = 0.99
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


def off_policy_baseline(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, rewards, states, rhos, actions, 
                        theta=0.9, delta=0.999, lr=0.01):

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
    return cum, std

def off_policy_traj_cv(env, behavior_policy, target_policy, num_episodes, max_steps_per_episode, rewards, states, rhos, actions, 
                       theta=0.9, delta=0.999, lr=0.01):
        
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
    return cum, std



def test_all_estimators(env, behavior_policy, target_policy_BASELINE, target_policy_REINFORCE, target_policy_DRPG, target_policy_TRAJECTORY,  
                        num_episodes, max_steps_per_episode, theta=0.9, delta = 0.999, lr=0.01):

    exp_rewards_DRPG = []
    exp_std_DRPG = []
    exp_rewards_REINFORCE = []
    exp_std_REINFORCE = []
    exp_rewards_baseline = []
    exp_std_baseline = []
    exp_rewards_traj_cv = []
    exp_std_traj_cv = []
      
    for episode in range(num_episodes):
        
        state = np.random.choice(env.num_states) 
        
        # sample trajectory using bahavior policy
        rewards          = []
        rhos_BASELINE    = []
        rhos_REINFORCE   = []
        rhos_DRPG        = []
        rhos_TRAJECTORY  = []
        states           = []
        actions          = []
        for t in range(max_steps_per_episode):
            action = behavior_policy.step(state)
            states.append(state)
            actions.append(action)
            next_state, reward = env.step(state, action)
            rewards.append(reward)
            rhos_BASELINE.append(target_policy_BASELINE.probs[state, action] / behavior_policy.probs[state, action])
            rhos_REINFORCE.append(target_policy_REINFORCE.probs[state, action] / behavior_policy.probs[state, action])
            rhos_DRPG.append(target_policy_DRPG.probs[state, action] / behavior_policy.probs[state, action])
            rhos_TRAJECTORY.append(target_policy_TRAJECTORY.probs[state, action] / behavior_policy.probs[state, action])
            state = next_state

        cum_DRPG, std_DRPG = off_policy_DRPG(env, behavior_policy, target_policy_DRPG, num_episodes, max_steps_per_episode, rewards, states, rhos_DRPG, actions)
        cum_REINFORCE, std_REINFORCE = off_policy_REINFORCE(env, behavior_policy, target_policy_REINFORCE, num_episodes, max_steps_per_episode, rewards, states, rhos_REINFORCE, actions)
        cum_baseline, std_baseline = off_policy_baseline(env, behavior_policy, target_policy_BASELINE, num_episodes, max_steps_per_episode, rewards, states, rhos_BASELINE, actions)
        cum_traj_cv, std_traj_cv = off_policy_traj_cv(env, behavior_policy, target_policy_TRAJECTORY, num_episodes, max_steps_per_episode, rewards, states, rhos_TRAJECTORY, actions)

        exp_rewards_DRPG.append(cum_DRPG)
        exp_std_DRPG.append(std_DRPG)
        exp_rewards_REINFORCE.append(cum_REINFORCE)
        exp_std_REINFORCE.append(std_REINFORCE)
        exp_rewards_baseline.append(cum_baseline)
        exp_std_baseline.append(std_baseline)
        exp_rewards_traj_cv.append(cum_traj_cv)
        exp_std_traj_cv.append(std_traj_cv)

    plt.plot(exp_rewards_DRPG, label='DRPG', color ='blue')
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards_DRPG)-1.96*np.array(exp_std_DRPG), np.array(exp_rewards_DRPG)+1.96*np.array(exp_std_DRPG), color='blue', alpha=0.2)
    plt.plot(exp_rewards_REINFORCE, label='REINFORCE', color='red')
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards_REINFORCE)-1.96*np.array(exp_std_REINFORCE), np.array(exp_rewards_REINFORCE)+1.96*np.array(exp_std_REINFORCE), color='red', alpha=0.2)
    plt.plot(exp_rewards_baseline, label='baseline', color='green')
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards_baseline)-1.96*np.array(exp_std_baseline), np.array(exp_rewards_baseline)+1.96*np.array(exp_std_baseline), color='green', alpha=0.2)
    plt.plot(exp_rewards_traj_cv, label='traj_cv', color='purple')
    plt.fill_between(np.arange(num_episodes), np.array(exp_rewards_traj_cv)-1.96*np.array(exp_std_traj_cv), np.array(exp_rewards_traj_cv)+1.96*np.array(exp_std_traj_cv), color='purple', alpha=0.2)

    plt.show()