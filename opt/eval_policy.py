import numpy as np
from pyrsistent import s
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *
from opt.mc_sim import *

def eval_policy_from_value_dic(sourcingEnv, value_dic, max_steps,
    max_stock = BIG_S,
    discount_fac = DISCOUNT_FAC,
    h_cost = H_COST, 
    b_penalty = B_PENALTY,
    n_visit_lim = 2,
    default_ss_policy = ss_policy_fastest_supp_backlog):

    sourcingEnv.reset()

    cost_sum = 0
    cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    for m in range(max_steps):
        possible_joint_actions = get_combo(int(max_stock - sourcingEnv.current_state.s), sourcingEnv.n_suppliers)
        max_q_value = -np.Inf
        best_action = np.zeros(sourcingEnv.n_suppliers) # order nothing is the supposed best action

        ss_action = default_ss_policy(sourcingEnv)
        q_value_ss = None

        for pa in possible_joint_actions:
            event_probs = sourcingEnv.get_event_probs(pa)
            value_contrib = 0
            reward_contrib = 0

            for e in range(len(event_probs)):
                event, supplier_index = sourcingEnv.get_event_tuple_from_index(e)

                potential_next_state = copy.deepcopy(sourcingEnv.current_state)
                if event == Event.DEMAND_ARRIVAL:
                    potential_next_state.s -= 1
                if event == Event.SUPPLY_ARRIVAL:
                    potential_next_state.s += potential_next_state.n_backorders[supplier_index]
                    potential_next_state.n_backorders[supplier_index] = 0
                    potential_next_state.n_backorders += pa
                if event == Event.SUPPLIER_ON:
                    potential_next_state.flag_on_off[supplier_index] = 1
                    potential_next_state.n_backorders += pa
                if event == Event.SUPPLIER_OFF:
                    potential_next_state.flag_on_off[supplier_index] = 0
                    potential_next_state.n_backorders += pa
                
                potential_immediate_cost = -cost_calc(potential_next_state, h_cost = h_cost, b_penalty = b_penalty)
                state_key = potential_next_state.get_repr_key()

                reward_contrib += event_probs[e] * potential_immediate_cost
                if state_key in value_dic:
                    if value_dic[state_key][1] > n_visit_lim:
                        potential_state_value = value_dic[state_key][0]
                    else:
                        sourcingEnvCopy = copy.deepcopy(sourcingEnv)
                        sourcingEnvCopy.current_state = potential_next_state
                        eval_costs = mc_with_ss_policy(sourcingEnvCopy,
                            periods = 10,
                            nested_mc_iters = 30)
                        potential_state_value = np.mean(eval_costs)
                    value_contrib += event_probs[e] * potential_state_value
            
            q_value = round(reward_contrib + discount_fac*value_contrib, 3)

            if q_value > max_q_value:
                max_q_value = q_value
                best_action_adp = pa
            
            # compute q value from sS policy action
            if (pa == ss_action).all():
                q_value_ss = q_value
        
        # determining action to take
        if q_value_ss == None:
            best_action = ss_action
        elif round(max_q_value, 1) > q_value_ss*2.5:
            best_action = best_action_adp
        else:
            best_action = ss_action
        
        sourcingEnv.step(ss_action)

        cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)

    return cost_sum



def eval_policy_from_ss_pol_2(sourcingEnv, value_dic, max_steps,
    max_stock = BIG_S,
    discount_fac = DISCOUNT_FAC,
    h_cost = H_COST, 
    b_penalty = B_PENALTY,
    default_ss_policy = ss_policy_fastest_supp_backlog):

    sourcingEnv.reset()

    cost_sum = 0
    cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    for m in range(max_steps):
        
        best_action = default_ss_policy(sourcingEnv)
        sourcingEnv.step(best_action)
        cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)

    return cost_sum


def eval_policy_from_ss_pol(sourcingEnv, value_dic, max_steps,
    discount_fac = DISCOUNT_FAC,
    h_cost = H_COST, 
    b_penalty = B_PENALTY,
    default_ss_policy = ss_policy_fastest_supp_backlog):

    sourcingEnv.reset()

    cost_sum = 0
    cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    for m in range(max_steps):
        ss_action = default_ss_policy(sourcingEnv)
        sourcingEnv.step(ss_action)
        cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)

    return cost_sum


def mc_eval_policy_perf(sourcingEnv, value_dic, max_steps = 1, mc_iters = 2,
    discount_fac = DISCOUNT_FAC,
    h_cost = H_COST, 
    b_penalty = B_PENALTY,
    policy_callback = eval_policy_from_value_dic):

    costs = []
    for mc in range(mc_iters):
        cost = policy_callback(sourcingEnv, value_dic, max_steps, discount_fac = discount_fac, h_cost = h_cost, b_penalty = b_penalty)
        costs.append(cost)
        print("MC eval iter: " + str(mc))
    
    return costs

