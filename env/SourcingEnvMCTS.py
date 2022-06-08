import numpy as np
from common.variables import *
from env.SourcingEnv import *
import itertools as it
from sim.sim_functions import *

class SourcingEnvMCTSWrapper():
    def __init__(self, sourcingEnv,
        max_episodes = PERIODS):

        self.sourcingEnv = sourcingEnv 

        self.n_steps = 0
        self.max_episodes = max_episodes
        self.action_size = self.sourcingEnv.action_size
    
    def getCurrentPlayer(self):
        return 1
    
    def getPossibleActions(self):
        self.single_sup_action_space = np.arange(0, self.action_size + 1, 1)
        self.joint_action_space_nested = [self.single_sup_action_space]*self.sourcingEnv.n_suppliers

        self.joint_action_space_list = [x for x in it.product(*self.joint_action_space_nested)]
        # self.joint_action_space = np.array(joint_action_space_list)

        return self.joint_action_space_list
    
    def takeAction(self, action_tuple):
        action = np.array(list(action_tuple))
        next_state, event, event_index, probs, supplier_index  = self.sourcingEnv.step(action)
        # self.n_steps += 1
        next_state_mcts = SourcingEnvMCTSWrapper(self.sourcingEnv)
        next_state_mcts.n_steps = self.n_steps + 1
        return next_state_mcts # Nasty
    
    def isTerminal(self):
        return self.n_steps > self.max_episodes
    
    def getReward(self):
        cost = -cost_calc(self.sourcingEnv.current_state)
        return cost

    
class Action():
    def __init__(self, x):
        self.x = x
        
    def __str__(self):
        return str(self.x, self.y)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self.x == self.other).all()

    def __hash__(self):
        return str(self.x)