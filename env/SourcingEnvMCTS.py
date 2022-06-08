import numpy as np
from common.variables import *
from env.SourcingEnv import *
import itertools as it
from sim.sim_functions import *

class SourcingEnvMCTSWrapper():
    def __init__(self, 
        order_quantity = 30,
        lambda_arrival = 10,
        procurement_cost_vec = np.array([2, 1.7]), 
        supplier_lead_times_vec = np.array([0.5, 0.75]), 
        on_times = np.array([3, 1]), 
        off_times = np.array([0.3, 1]),
        ):

        self.sourcingEnv = sourcingEnv = SourcingEnv(
            order_quantity = order_quantity,
            lambda_arrival = lambda_arrival, # or 10
            procurement_cost_vec = procurement_cost_vec,
            supplier_lead_times_vec = supplier_lead_times_vec,
            on_times = on_times, 
            off_times = off_times
        )

        self.n_steps = 0
        self.max_episodes = self.sourcingEnv.max_episodes
    
    def getCurrentPlayer(self):
        return 1
    
    def getPossibleActions(self):
        self.single_sup_action_space = np.arange(0, self.sourcingEnv.action_size + 1, 1)
        self.joint_action_space_nested = [self.single_sup_action_space]*self.sourcingEnv.n_suppliers

        self.joint_action_space_list = [x for x in it.product(*self.joint_action_space_nested)]
        # self.joint_action_space = np.array(joint_action_space_list)

        return self.joint_action_space_list
    
    def takeAction(self, action_tuple):
        action = np.array(list(action_tuple))
        next_state, event, event_index, probs, supplier_index  = self.sourcingEnv.step(action)
        self.n_steps += 1
        return next_state
    
    def isTerminal(self):
        # return self.n_steps > self.max_episodes
        return self.sourcingEnv.isTerminal()
    
    def getReward(self):
        return -cost_calc(self.sourcingEnv.current_state)

    
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