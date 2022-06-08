from enum import Enum
import numpy as np
from common.variables import *
import itertools as it
from sim.sim_functions import *


# indexed 0,1,2,3...
class Event(Enum):
    DEMAND_ARRIVAL = 0
    SUPPLY_ARRIVAL = 1
    SUPPLIER_ON = 2
    SUPPLIER_OFF = 3
    NO_EVENT = 4

# indexed 0,1,2,3...
# indexed 0,1,2,3...
class MState():

    def __init__(self, sourcingEnv,
        stock_level = 0, 
        n_suppliers = 2,
        n_backorders = False,
        flag_on_off = False,
        ):
        
        self.s = stock_level # stock level
        self.n_backorders = n_backorders if isinstance(n_backorders, np.ndarray) else np.zeros(n_suppliers)
        self.flag_on_off = flag_on_off if isinstance(flag_on_off, np.ndarray) else np.ones(n_suppliers)
        # self.flag_on_off = np.ones(n_suppliers) if not flag_on_off.any() else flag_on_off # on off flag

        # superfluous mcts
        self.sourcingEnv = sourcingEnv
        self.action_size = self.sourcingEnv.action_size
        self.n_suppliers = n_suppliers
    
    def __str__(self):
        return "Stock: {fname}, n backorders: {nb}, supplier status (on/off): {sup_stat}".format(fname = self.s, nb = self.n_backorders, sup_stat = self.flag_on_off)
    
    # superfluous mcts
    def isTerminal(self):
        return self.sourcingEnv.isTerminal()
    
    def getCurrentPlayer(self):
        return 1
    
    def getPossibleActions(self):
        self.single_sup_action_space = np.arange(0, self.action_size + 1, 1)
        self.joint_action_space_nested = [self.single_sup_action_space]*self.n_suppliers

        # now slow as shit
        self.joint_action_space_list = [x for x in it.product(*self.joint_action_space_nested)]

        return self.joint_action_space_list

    def takeAction(self, action_tuple):
        action = np.array(list(action_tuple))
        next_state, event, event_index, probs, supplier_index  = self.sourcingEnv.step(action)
        return next_state
    
    def getReward(self):
        return -cost_calc(self.sourcingEnv.current_state)




