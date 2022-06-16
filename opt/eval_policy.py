import numpy as np
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *

def eval_policy_from_value_dic(sourcingEnv, value_dic, 
    max_steps = MAX_STEPS,
    max_stock = BIG_S,
    h_cost = H_COST, 
    b_penalty = B_PENALTY):

    sourcingEnv.reset()

    cost_sum = 0
    cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    for m in range(max_steps):
        possible_joint_actions = get_combo(int(max_stock - sourcingEnv.current_state.s), sourcingEnv.n_suppliers)

        state = sourcingEnv.current_state
        state_key = state.get_repr_key()





