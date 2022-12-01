import numpy as np
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *
from opt.mc_sim import *
from tqdm import tqdm


def eval_policy_from_value_dic(sourcingEnv, 
        policy = ss_policy_fastest_supp_backlog, 
        **kwargs
    ):

    # value_dic, max_steps,
    #     sub_nested_mc_iter = SUB_NESTED_MC_ITER

    max_stock = BIG_S if "max_stock" not in kwargs else kwargs["max_stock"]
    periods = PERIODS if "periods_val_it" not in kwargs else kwargs["periods_val_it"]

    b_penalty = B_PENALTY if "b_penalty" not in kwargs else kwargs["b_penalty"]
    h_cost = H_COST if "h_cost" not in kwargs else kwargs["h_cost"]
    
    default_ss_policy = ss_policy_fastest_supp_backlog if "default_ss_policy" not in kwargs else kwargs["default_ss_policy"]
    n_visit_lim = N_VISIT_LIM if "n_visit_lim" not in kwargs else kwargs["n_visit_lim"]
    discount_fac = DISCOUNT_FAC if "discount_fac" not in kwargs else kwargs["discount_fac"]
    safe_factor = SAFE_FACTOR if "safe_factor" not in kwargs else kwargs["safe_factor"]
    sub_eval_periods = SUB_EVAL_PERIODS if "sub_eval_periods" not in kwargs else kwargs["sub_eval_periods"]
    sub_nested_mc_iter = SUB_NESTED_MC_ITER if "sub_nested_mc_iter" not in kwargs else kwargs["sub_nested_mc_iter"]
    value_dic = None if "value_dic" not in kwargs else kwargs["value_dic"]
    if value_dic == None:
        print("Error no value dic supplied!")
        return False
    
    approx_eval = False if "approx_eval" not in kwargs else True
    
    # sourcingEnv.reset()

    cost_sum = 0
    
    # for m in tqdm(range(periods)):
    for m in range(periods):
        cost_sum += cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)

        possible_joint_actions = get_combo(int(max_stock - sourcingEnv.current_state.s), sourcingEnv.n_suppliers) if not approx_eval else get_combo_reduction(int(max_stock - sourcingEnv.current_state.s), sourcingEnv.n_suppliers)
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
                
                potential_next_cost = -cost_calc(potential_next_state, h_cost = h_cost, b_penalty = b_penalty)
                state_key = potential_next_state.get_repr_key()
                
                if hasattr(sourcingEnv, 'fixed_costs)'):
                    fixed_costs = get_fixed_costs(pa, fixed_costs_vec = sourcingEnv.fixed_costs)
                else:
                    fixed_costs = [0]*sourcingEnv.n_suppliers
                
                potential_immediate_cost = -np.sum(np.multiply(sourcingEnv.procurement_cost_vec, pa)) -np.sum(fixed_costs)

                reward_contrib += event_probs[e] * (potential_next_cost + potential_immediate_cost)
                
                if state_key in value_dic and value_dic[state_key][1] > n_visit_lim:
                        potential_state_value = value_dic[state_key][0]
                else:
                    sourcingEnvCopy = copy.deepcopy(sourcingEnv)
                    sourcingEnvCopy.current_state = potential_next_state
                    eval_costs = mc_with_policy(sourcingEnvCopy,
                        periods = sub_eval_periods,
                        nested_mc_iters = sub_nested_mc_iter,
                        policy_callback = default_ss_policy)
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
        sf = 1/safe_factor if q_value_ss is not None and q_value_ss < 0 else safe_factor
        if q_value_ss == None:
            best_action = ss_action
        elif round(max_q_value, 1) > q_value_ss*sf:
            best_action = best_action_adp
        else:
            best_action = ss_action
        
        # sourcingEnv.step(best_action)
        # total_procurement_cost = np.sum(np.multiply(best_action, sourcingEnv.procurement_cost_vec))
        # cost_sum += total_procurement_cost
        
    # sourcingEnv.reset()
    return best_action


def lp_mdp_policy(sourcingEnv, **kwargs):

    filename = None if "filename" not in kwargs else kwargs["filename"]
    if filename == None:
        print("No LP file detected!")
        return False

    filename_lp = "output/lp_sol_" + filename
    with open(filename_lp, 'rb') as f:
        lp_sol = pkl.load(f)
    
    with open("output/" + filename, 'rb') as f:
        output_obj = pkl.load(f)
        
    lp_strs = [x[0] for x in lp_sol]
    lp_strs_tups = [(x.split("..")[1], x.split("..")[2]) for x in lp_strs]
    lp_tups = [(x[0], [int(s) for s in re.findall(r'\d+', x[1])] ) for x in lp_strs_tups]

    lp_dic = dict(lp_tups)

    state_key = sourcingEnv.current_state.get_nested_list_repr()

    if state_key in lp_dic and np.sum(lp_dic[state_key]) != 0:
        order_vec = np.array(lp_dic[state_key])
        return order_vec
    else:
        order_vec = eval_policy_from_value_dic(sourcingEnv, **kwargs)

    return order_vec