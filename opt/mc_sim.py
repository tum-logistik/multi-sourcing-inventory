import numpy as np
from env.SourcingEnv import *
from sim.sim_functions import *
from sim.policies import *

def mc_with_ss_policy(sourcingEnv, h_cost = 4, b_penalty = 6, small_s = 3, big_s = 10, periods = PERIODS):

        cost = cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
        total_costs = [cost]
        for i in range(periods):
            
            policy_action = ss_policy_rand_supp_backlog(sourcingEnv, small_s = small_s, big_s = big_s)
            next_state, event, event_index, probs, supplier_index = sourcingEnv.step(policy_action)
            cost = cost_calc(next_state, h_cost = h_cost, b_penalty = b_penalty)
            total_costs.append(cost)

        return np.sum(total_costs), np.sum(total_costs)/periods