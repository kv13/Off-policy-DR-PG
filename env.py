import numpy as np

class Tabular_MDP:

    def __init__(self, num_states, num_rows, num_cols, num_actions, deterministic, transition_probs = [], gamma=1):
        self.num_states    = num_states
        self.num_rows      = num_rows
        self.num_cols      = num_cols
        self.num_actions   = num_actions
        self.deterministic = deterministic
        self.P             = transition_probs
        self.gamma         = gamma
    
    def _init_rewards_random(self):
        # r(s,a) is a function of pair state, action
        self.R = np.random.rand(self.num_states, self.num_actions)
    
    def _init_rewards_right_up_corner(self):
        # r(s,a) -1 for each pair state,action except right up corner
        self.R = -1 * np.ones((self.num_states, self.num_actions))
        self.R[self.num_cols-1, 1] = 2
        self.R[self.num_cols-1, 2] = 2
    
    def _init_rewards_easy(self):

        off_grid_reward = -2
        goal_reward     =  2
        self.R          = -np.ones((self.num_states, self.num_actions))
        goal_state = self.num_states - 1 # bottom left
        
        for i in range(self.num_rows):
            self.R[i * self.num_cols, 0]       = off_grid_reward
            self.R[(i+1) * self.num_cols-1, 2] = off_grid_reward
        
        for i in range(self.num_cols):
            self.R[i, 1] = off_grid_reward
            self.R[(self.num_rows-1)*self.num_cols+i, 3] = off_grid_reward
        
        self.R[goal_state, 2] = goal_reward
        self.R[goal_state, 3] = goal_reward
        print('rewards', self.R)

    def step(self, state, action):
        # if agent hits the boundary remains at the same place    
        if self.deterministic == True:
            if action == 0:
                next_state = state-1
                if state % self.num_cols == 0:
                    next_state = state
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
                raise ValueError("step function went wrong.")
        else:
            # TODO: implement correctly the dynamics. STATUS: not working
            print("step: SHOULD NOT ENTER HERE...")
            next_state = np.random.choice(self.num_states, p=self.P[state, action])
         
        reward = self.R[state, action]
        return next_state, reward