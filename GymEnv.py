from opt.mc_sim import *
from common.variables import *
from sim.sim_functions import *
import torch
import gym
from gym import spaces
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple, MultiBinary

s = SourcingEnv()


class CustomGymEnv(Env):

    def __init__(self, sourcing_env):
            self.SourcingEnv = sourcing_env
            self.counter = 0
    
            # Actions we can take
            self.action_space = MultiDiscrete([INVEN_LIMIT,INVEN_LIMIT])
            
            # Inventory Observation State
            self.observation_space = Box(low=np.array([-30, 0, 0, 0, 0]), high=np.array([30, 30, 30, 1, 1]), shape=(5,), dtype=int)
                                                   
        
    def step(self, action):
                                            
            next_state, event, i, event_probs, supplier_index = self.SourcingEnv.step(action)
            self.counter += 1
            
            if (next_state.s) >=0: 
                reward = -float((2 * next_state.s) + np.sum(np.multiply(action, PROCUREMENT_COST_VEC)))
            else: 
                reward = -float((-10* next_state.s) + np.sum(np.multiply(action, PROCUREMENT_COST_VEC)))
            
            info = {}
            
            if self.counter < PERIODS:
                done = False
            else:
                done = True
            
            next_state_array = np.array(next_state.get_list_repr())
            return next_state_array, reward, done, info
        
    def reset(self):
            self.SourcingEnv = SourcingEnv()
            return np.array(self.SourcingEnv.current_state.get_list_repr())
   
 