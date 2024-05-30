import numpy as np

class Policy:  

    def __init__(self, num_states, num_actions):
        self.num_states  = num_states
        self.num_actions = num_actions
    
    def _init_unif(self): 
        # why thetas are init in 1???   
        self.thetas      = np.zeros((self.num_states, self.num_actions))
        self.probs       = np.zeros((self.num_states, self.num_actions))
        for state in range(self.num_states):
            self.probs[state, :] = np.exp(self.thetas[state, :]) / sum(np.exp(self.thetas[state, :]))  
    
    def _init_rand(self,alpha):
        # init thetas around 0???
        self.thetas = (1-alpha) * np.zeros((self.num_states, self.num_actions)) + alpha * (2 * np.random.rand(self.num_states, self.num_actions) -1)
        self.probs  = np.zeros((self.num_states, self.num_actions))
        for state in range(self.num_states):
            self.probs[state, :] = np.exp(self.thetas[state, :]) / sum(np.exp(self.thetas[state, :]))

    def step(self, state):
        next_action =  np.random.choice(self.num_actions, p=self.probs[state, :])
        return next_action