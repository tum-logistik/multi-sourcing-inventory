import numpy as np
from common.variables import *

# simulation functions

def cost_calc(state, h_cost = H_COST, b_penalty = B_PENALTY):
    cost = state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)
    return cost

# TODO: Create an expected cost function
def cost_calc_expected_di(sourcingEnv, order_quantity_vec, h_cost = H_COST, b_penalty = B_PENALTY):
    # TODO: Incorporate discount factor

    event_probs = sourcingEnv.get_event_probs(order_quantity_vec)
    current_cost = cost_calc(sourcingEnv.current_state)
    proc_costs_pro_rata = sourcingEnv.procurement_cost_vec
    total_proc_costs = np.sum(np.multiply(proc_costs_pro_rata, order_quantity_vec))

    exp_demand_cost = event_probs[0] * (current_cost - h_cost if sourcingEnv.current_state.s - 1 >= 0  else current_cost + b_penalty)
    exp_hold_cost_0 = event_probs[1] * current_cost * h_cost * order_quantity_vec[0]
    exp_hold_cost_1 = event_probs[2] * current_cost * h_cost * order_quantity_vec[1]
    exp_other_costs = (1 - np.sum(event_probs[0:3])) * current_cost

    exp_total_cost = total_proc_costs + exp_demand_cost + exp_hold_cost_0 + exp_hold_cost_1 + exp_other_costs

    # cost = state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)
    return exp_total_cost

def get_combo(y, n):
    return np.array(np.meshgrid(*[range(0, y) for x in range(n)])).T.reshape(-1, n)