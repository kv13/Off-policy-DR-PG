import env
import policy
import numpy as np
import off_policy as ofp

def main():
    num_rows     = 3
    num_cols     = 5
    num_states   = num_cols * num_rows
    num_actions  = 4 # 0 = left, 1 = up, 2 = right, 3 = down
    num_episodes = 100
    T            = 50
    # deterministic env or stochastic env
    deterministic = True

    # create enviroment
    envtab = env.Tabular_MDP(num_states, num_rows, num_cols, num_actions, deterministic)
    envtab._init_rewards_easy()
        
    # policies
    behavior_policy  = policy.Policy(num_states, num_actions)
    behavior_policy._init_unif()

    target_policy    = policy.Policy(num_states, num_actions)
    target_policy._init_rand(0.05)

    # _ = ofp.off_policy_REINFORCE(envtab, 
    #                              behavior_policy, 
    #                              target_policy, 
    #                              num_episodes,
    #                              T) 
    
    _ = ofp.off_policy_DRPG(envtab, 
                            behavior_policy, 
                            target_policy, 
                            num_episodes,
                            T)
    
if __name__ == '__main__':
    main()