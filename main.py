import numpy as np
import off_policy_DRPG


def main():
    num_rows     = 3
    num_cols     = 5
    num_states   = num_cols * num_rows
    num_actions  = 4
    num_episodes = 1
    T            = 20
    
    # define rewards 
    rewards     = np.random.rand(num_states, num_actions)
    
    # define transition probabilities 
    # deterministic / stochastic

    deterministic    = True
    if deterministic == False:
        # TO DO
        #transition_probs = utils.create_trans_matrix(deterministic, num_states, num_actions)
        pass
    
    # create enviroment
    env = off_policy_DRPG.Tabular_MDP(num_states, num_rows, num_cols, num_actions, rewards, deterministic)
    
    # policies
    behavior_policy = off_policy_DRPG.Policy(num_states, num_actions)
    target_policy    = off_policy_DRPG.Policy(num_states, num_actions)

    _ = off_policy_DRPG.off_policy_DRPG(env, 
                                        behavior_policy, 
                                        target_policy, 
                                        num_episodes, 
                                        T)
    
if __name__ == '__main__':
    main()