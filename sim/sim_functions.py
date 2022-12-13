import numpy as np
from common.variables import *

# simulation functions

def cost_calc(state, h_cost = H_COST, b_penalty = B_PENALTY):
    holding_cost = state.s * h_cost
    bo_penalty = np.abs(state.s * b_penalty)
    unit_cost = holding_cost if state.s > 0 else bo_penalty
    cost = unit_cost * state.state_tau
    # cost = (state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)) * state.state_tau
    return cost

def cost_calc_expected_di(sourcingEnv, order_quantity_vec, custom_state = None, h_cost = H_COST, b_penalty = B_PENALTY):

    if custom_state is None:
        custom_state = sourcingEnv.current_state

    event_probs = sourcingEnv.get_event_probs(order_quantity_vec)
    current_cost = cost_calc(custom_state)

    proc_costs_pro_rata = sourcingEnv.procurement_cost_vec
    procurement_costs_avail = np.multiply(proc_costs_pro_rata, sourcingEnv.current_state.flag_on_off)

    if hasattr(sourcingEnv, 'fixed_costs)'):
        fixed_costs = get_fixed_costs(order_quantity_vec, fixed_costs_vec = sourcingEnv.fixed_costs)
    else:
        fixed_costs = [0]*sourcingEnv.n_suppliers
    
    total_proc_costs = np.sum(np.multiply(procurement_costs_avail, order_quantity_vec)) + np.sum(fixed_costs)

    tau = sourcingEnv.current_state.state_tau
    exp_demand_cost = event_probs[0] * ((custom_state.s-1) * h_cost if custom_state.s - 1 >= 0  else np.abs(custom_state.s-1) * b_penalty)
    exp_hold_cost_0 = event_probs[1] * ((custom_state.s + order_quantity_vec[0]) * h_cost if custom_state.s + order_quantity_vec[0] >= 0  else (custom_state.s + order_quantity_vec[0]) * b_penalty)
    exp_hold_cost_1 = event_probs[2] * ((custom_state.s + order_quantity_vec[1]) * h_cost if custom_state.s + order_quantity_vec[0] >= 0  else (custom_state.s + order_quantity_vec[1]) * b_penalty)
    exp_other_costs = (1 - np.sum(event_probs[0:3])) * current_cost

    exp_total_cost = total_proc_costs + (exp_demand_cost + exp_hold_cost_0 + exp_hold_cost_1 + exp_other_costs) * tau

    # cost = state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)
    return exp_total_cost

def get_fixed_costs(order_vec, fixed_costs_vec = FIXED_COST_VEC):
    fixed_costs_binary = [int(x > 0) for x in order_vec]
    fixed_costs = np.multiply(fixed_costs_binary, fixed_costs_vec)
    return fixed_costs

def get_combo(y, n):
    return np.array(np.meshgrid(*[range(0, y) for x in range(n)])).T.reshape(-1, n)

def get_combo_reduction(y, n):
    # an sS constrained search
    full_combo = np.array(np.meshgrid(*[range(0, y) for x in range(n)])).T.reshape(-1, n)
    filtered_output = np.array(list(filter(lambda x: 0 in x, list(full_combo))))
    return filtered_output