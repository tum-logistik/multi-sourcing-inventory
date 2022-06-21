import numpy as np
from pyrsistent import s
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *


def eval_policy_from_value_dic(sourcingEnv, value_dic, max_steps,
    max_stock = BIG_S,
    discount_fac = DISCOUNT_FAC,
    h_cost = H_COST, 
    b_penalty = B_PENALTY):

    sourcingEnv.reset()

    cost_sum = 0
    cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    for m in range(max_steps):
        possible_joint_actions = get_combo(int(max_stock - sourcingEnv.current_state.s), sourcingEnv.n_suppliers)
        max_q_value = -np.Inf
        best_action = np.zeros(sourcingEnv.n_suppliers) # order nothing is the supposed best action

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
                    potential_state_value = value_dic[state_key][0]

                    value_contrib += event_probs[e] * potential_state_value

            q_value = round(reward_contrib + discount_fac*value_contrib, 3)

            if q_value > max_q_value:
                max_q_value = q_value
                best_action = pa

        sourcingEnv.step(best_action)

        cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)

    return cost_sum


def mc_eval_policy_from_value_dic(sourcingEnv, value_dic, max_steps = 1, mc_iters = 2,
    max_stock = BIG_S,
    discount_fac = DISCOUNT_FAC,
    h_cost = H_COST, 
    b_penalty = B_PENALTY):

    costs = []
    for mc in range(mc_iters):
        cost = eval_policy_from_value_dic(sourcingEnv, value_dic, max_steps, 
            max_stock = max_stock, discount_fac = discount_fac, h_cost = h_cost, b_penalty = b_penalty)
        costs.append(cost)
        print("MC eval iter: " + str(mc))
    
    return costs