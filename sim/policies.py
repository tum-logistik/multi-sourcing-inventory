import numpy as np
from common.variables import *
from scipy.stats import poisson
from sim.sim_functions import *
import copy
from env.HelperClasses import *
import time
from tqdm import tqdm
import pickle as pkl
import re
# from opt.mc_sim import *
# from opt.eval_policy import *


def ss_policy(inventory_level, small_s, big_s):

    assert big_s > small_s, "Assertion Failed: Big S smaller than small S"

    if inventory_level < small_s:
        return big_s - inventory_level
    else:
        return 0

# order from quickest available supplier
def ss_policy_fastest_supp_backlog(sourcingEnv, **kwargs):
    small_s = SMALL_S if "small_s" not in kwargs else kwargs["small_s"]
    big_s = BIG_S if "big_s" not in kwargs else kwargs["big_s"]
    
    ss_pol_suggest = ss_policy(sourcingEnv.current_state.s, small_s = small_s, big_s = big_s) 
    total_order_amount = np.clip(ss_pol_suggest - np.sum(sourcingEnv.current_state.n_backorders), 0, big_s)
    
    avail_mu_lt_prod = sourcingEnv.mu_lt_rate * sourcingEnv.current_state.flag_on_off
    supp_index = np.argmax(avail_mu_lt_prod)

    policy_action = np.zeros(sourcingEnv.n_suppliers)
    policy_action[supp_index] = total_order_amount
    
    return policy_action


# implement dual index policy
def dual_index_policy(sourcingEnv, **kwargs):

    h_cost = H_COST if "h_cost" not in kwargs else kwargs["h_cost"]
    b_penalty = B_PENALTY if "b_penalty" not in kwargs else kwargs["b_penalty"]
    delta_cand_range = DI_DEL_RNG if "delta_cand_range" not in kwargs else kwargs["delta_cand_range"]
    safety_factor_di = DI_SF_FAC if "safety_factor_di" not in kwargs else kwargs["safety_factor_di"]
    max_order = BIG_S if "max_order" not in kwargs else kwargs["max_order"]

    assert sourcingEnv.tracking_flag, "Assertion: Tracking feature must be on for dual index policy"

    ord_vec = np.zeros(len(sourcingEnv.supplier_lead_times_vec))
    # print("Opt vec: " + str(ord_vec_opt))
    # Safety cap
    if sourcingEnv.current_state.s >= safety_factor_di: # and False:
        # kwargs1 = {"periods" : 30,
        #     "nested_mc_iters" : 30,
        #     "supplier_index": 1
        # }
        # ord_vec_opt = single_source_orderupto_policy(sourcingEnv, **kwargs1)
        ord_vec_opt = ss_policy_fastest_supp_backlog(sourcingEnv)
    elif sourcingEnv.current_state.s < 0:# and False:
        kwargs1 = {"periods" : 30,
            "nested_mc_iters" : 30,
            "supplier_index": 1
        }
        ord_vec_opt = single_source_orderupto_policy(sourcingEnv, **kwargs1)
        # ord_vec_opt = ss_policy_fastest_supp_backlog(sourcingEnv)
    # elif False:
    else:
        min_cost = np.Inf
        ze_opt = 0
        zr_opt = 0

        tmark_exp = sourcingEnv.get_time_mark(sourcingEnv.action_history_tuple, sourcingEnv.exp_ind)
        tmark_reg = sourcingEnv.get_time_mark(sourcingEnv.action_history_tuple, sourcingEnv.reg_ind)

        overshoot_range = np.array([x for x in sourcingEnv.action_history_tuple if tmark_reg < x[0] < tmark_exp])
        action_history_ov_range = np.array([x[1] for x in overshoot_range])

        reg_orders_ov_range = action_history_ov_range[:, sourcingEnv.reg_ind] if len(action_history_ov_range) > 0 else np.array([])
        cum_reg_orders_ov_range = np.sum(reg_orders_ov_range)

        exp_orders_ov_range = action_history_ov_range[:, sourcingEnv.exp_ind] if len(action_history_ov_range) > 0 else np.array([])
        cum_exp_orders_ov_range = np.sum(exp_orders_ov_range)

        cf = b_penalty / (b_penalty + h_cost)
        perc_cand = np.clip(0, cf, 1.0)

        delta_calc, gap = inv_poisson(perc_cand, sourcingEnv.lambda_arrival)

        for ze in range(0-DI_SF_FAC, DI_DEL_RNG):
            for delt in range(0-DI_SF_FAC, delta_calc + DI_SF_FAC):
                zr = ze + int(delt)
                ord_vec[sourcingEnv.exp_ind] = np.clip(ze - cum_reg_orders_ov_range, 0, np.Inf)
                ord_vec[sourcingEnv.reg_ind] = np.clip(zr - cum_exp_orders_ov_range, 0, np.Inf)

                delta_cost = cost_calc_expected_di(sourcingEnv, ord_vec)
                if delta_cost < min_cost:
                    min_cost = delta_cost
                    ze_opt = ze
                    zr_opt = zr
                    ord_vec_opt = ord_vec
                
        ord_vec_opt[sourcingEnv.exp_ind] = np.clip(ze_opt - sourcingEnv.current_state.s, 0, np.Inf)
        ord_vec_opt[sourcingEnv.reg_ind] = np.clip(zr_opt - sourcingEnv.current_state.s, 0, np.Inf)
    
    ord_vec_opt_clip = np.clip(ord_vec_opt, 0, max_order)

    return ord_vec_opt_clip


def inv_poisson(perc, lambda_arrival, x_lim = 60, delt = 0):
    gap = np.Inf
    x_opt = 0
    for x in range(x_lim):
        perc_cand = poisson.cdf(x + delt, lambda_arrival)
        if np.abs(perc - perc_cand) < gap:
            gap = np.abs(perc - perc_cand)
            x_opt = x
    
    return x_opt, gap

# implement single supplier newsvendor,
def newsvendor_opt_order(procurement_cost, b = B_PENALTY, h = H_COST, lambda_arrival = LAMBDA):
    cf = np.clip((b - procurement_cost) / (b + h), 0, 1)
    # opt_inventory = inv_poisson(cf, lambda_arrival = lambda_arrival)
    opt_inventory = np.clip(poisson.ppf(cf, lambda_arrival), 0, np.Inf)
    return opt_inventory

def single_source_orderupto_policy(sourcingEnv, **kwargs):
    supplier_index = 0 if "supplier_index" not in kwargs else kwargs["supplier_index"]
    procurement_cost_vec = PROCUREMENT_COST_VEC if "procurement_cost_vec" not in kwargs else kwargs["procurement_cost_vec"]
    
    fixed_cost = sourcingEnv.fixed_costs[supplier_index] if hasattr(sourcingEnv, 'fixed_costs)') else 0
    procurement_cost = procurement_cost_vec[supplier_index] + fixed_cost
    opt_inventory = newsvendor_opt_order(procurement_cost)
    order_amount = np.clip(opt_inventory - sourcingEnv.current_state.s, 0, np.Inf)

    action_array = np.zeros(sourcingEnv.n_suppliers)
    action_array[supplier_index] = order_amount
    return action_array

# implement kiesmueller heuristic, / COP


# Myopic 1 policy, 1 step ahead myopic policy.
def myopic1_policy(sourcingEnv, **kwargs):

    max_order = MAX_INVEN if "max_order" not in kwargs else kwargs["max_order"]

    possible_joint_actions = get_combo(int(max_order), sourcingEnv.n_suppliers)
    myopic_order = np.zeros(sourcingEnv.n_suppliers)

    cost = np.Inf
    for a in possible_joint_actions:
        cand_cost = cost_calc_expected_di(sourcingEnv, a)
        if cand_cost < cost:
            myopic_order = a
            cost = cand_cost
    
    return myopic_order

def myopic2_policy(sourcingEnv, **kwargs):

    max_order = BIG_S if "max_order" not in kwargs else kwargs["max_order"]

    possible_joint_actions = get_combo(int(max_order), sourcingEnv.n_suppliers)
    myopic_order = np.zeros(sourcingEnv.n_suppliers)

    cost = np.Inf
    for a in possible_joint_actions:
        cand_cost = cost_calc_expected_di(sourcingEnv, a)
        sourcingEnvGhost = copy.deepcopy(sourcingEnv)
        for event in [(Event.DEMAND_ARRIVAL, None), (Event.SUPPLY_ARRIVAL, 0), (Event.SUPPLY_ARRIVAL, 1), (Event.SUPPLIER_OFF, 0), (Event.SUPPLIER_OFF, 1) ]:
            supplier = event[1]
            _, _, i, event_probs, _ = sourcingEnvGhost.step(a, (event[0], supplier))
            prob_event = event_probs[i]
            for a2 in possible_joint_actions:
                stage2_cost = prob_event*cost_calc_expected_di(sourcingEnvGhost, a2)
                cand_cost += stage2_cost
        
        if cand_cost < cost:
            myopic_order = a
            cost = cand_cost
    
    return myopic_order



def best_single_supplier_index(sourcingEnv):

    single_supplier_mean_costs = []
    sing_supp_mean_cost = np.Inf
    for s in range(sourcingEnv.n_suppliers):
        kwargs = {"periods" : 30,
            "nested_mc_iters" : 100,
            "h_cost": H_COST,
            "b_penalty": B_PENALTY,
            "supplier_index": s
        }

        single_supplier_costs = mc_with_policy(sourcingEnv, start_state = sourcingEnv.current_state, 
            use_tqdm = True,
            policy_callback = single_source_orderupto_policy,
            **kwargs)
        
        sing_supp_mean_cost_i = np.mean(single_supplier_costs)
        single_supplier_mean_costs.append(sing_supp_mean_cost_i)

        if sing_supp_mean_cost_i < sing_supp_mean_cost:
            single_supplier_costs_select = single_supplier_costs
            sing_supp_mean_cost = sing_supp_mean_cost_i
    
    return np.argmin(single_supplier_mean_costs)
### Legacy
# orders up to S items, when inven < s, orders with a uniform random supplier, order amount S - s
def ss_policy_rand_supp(sourcingEnv, small_s, big_s):

    total_order_amount = ss_policy(sourcingEnv.current_state.s, small_s = small_s, big_s = big_s)
    random_supplier_index = np.random.randint(0, sourcingEnv.n_suppliers)
    policy_action = np.zeros(sourcingEnv.n_suppliers)
    policy_action[random_supplier_index] = total_order_amount
    
    return policy_action

# orders up to S items, when inven < s, orders with a uniform random supplier, order amount S - sum(backlog)
def ss_policy_rand_supp_backlog(sourcingEnv, small_s, big_s):

    ss_pol_suggest = ss_policy(sourcingEnv.current_state.s, small_s = small_s, big_s = big_s) 
    total_order_amount = np.clip(ss_pol_suggest- np.sum(sourcingEnv.current_state.n_backorders), 0, big_s)

    random_supplier_index = np.random.randint(0, sourcingEnv.n_suppliers)
    policy_action = np.zeros(sourcingEnv.n_suppliers)
    policy_action[random_supplier_index] = total_order_amount
    
    return policy_action

# SSN, best single sourcing newsvendor solution
def ssn_policy(sourcingEnv, **kwargs):

    s_custom = MState(stock_level = 0, 
        n_suppliers = N_SUPPLIERS, 
        n_backorders = np.array([0, 0]), 
        flag_on_off = np.array([1, 1]))

    opt_cost = np.Inf
    order_vec = np.zeros(sourcingEnv.n_suppliers)
    
    for s in range(sourcingEnv.n_suppliers):
        single_supplier_costs = mc_with_policy(sourcingEnv, start_state = s_custom, 
            use_tqdm = False,
            policy_callback = single_source_orderupto_policy,
            **kwargs)
        ssup_cost = np.min(single_supplier_costs)
        if ssup_cost < opt_cost:
            order_vec = np.zeros(sourcingEnv.n_suppliers)
            order_vec[s] = np.array(newsvendor_opt_order(sourcingEnv.procurement_cost_vec[s]))

            # order_action = single_source_orderupto_policy(sourcingEnv, **kwargs)
            # order_vec = order_action
    
    return order_vec

def ssn_policy_fast(sourcingEnv, **kwargs):
    cost = np.Inf
    order_vec = np.zeros(sourcingEnv.n_suppliers)
    for s in range(sourcingEnv.n_suppliers):
        order_vec_cand = np.zeros(sourcingEnv.n_suppliers)
        order_vec_cand[s] = np.array(newsvendor_opt_order(sourcingEnv.procurement_cost_vec[s]))
        fixed_costs = get_fixed_costs(sourcingEnv.fixed_costs, fixed_cost_vec = sourcingEnv.fixed_costs)
        cost_cand = cost_calc(sourcingEnv.current_state) + sourcingEnv.procurement_cost_vec[s]*order_vec_cand[s] + fixed_costs[s]
        if cost_cand < cost:
            cost = cost_cand
            order_vec = order_vec_cand
    return order_vec

def mc_episode_with_policy(sourcingEnv, 
    policy = ss_policy_fastest_supp_backlog, 
    **kwargs):

    b_penalty = B_PENALTY if "b_penalty" not in kwargs else kwargs["b_penalty"]
    h_cost = H_COST if "h_cost" not in kwargs else kwargs["h_cost"]
    periods = PERIODS if "periods" not in kwargs else kwargs["periods"]
    norm_m_flag = True if "norm_m_flag" not in kwargs else kwargs["norm_m_flag"]
    debug_flag = False if "debug_flag" not in kwargs else kwargs["debug_flag"]
    discount_fac = DISCOUNT_FAC if "discount_fac" not in kwargs else kwargs["discount_fac"]
    
    sourcingEnv.reset()

    # cost = cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    total_costs = []
    tau_sum = 0.0
    
    if debug_flag:
        event_tracker = []
    
    for i in range(periods):
        
        policy_action = policy(sourcingEnv, **kwargs)
        
        cost = cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)

        if hasattr(sourcingEnv, 'fixed_costs'):
            fixed_costs = get_fixed_costs(policy_action, fixed_costs_vec = sourcingEnv.fixed_costs)
        else:
            fixed_costs = [0]*sourcingEnv.n_suppliers
        
        procurement_cost_if_avail = np.multiply(policy_action, sourcingEnv.procurement_cost_vec)
        procurement_cost = np.sum(np.multiply(procurement_cost_if_avail, sourcingEnv.current_state.flag_on_off))

        total_procurement_cost = procurement_cost + np.sum(fixed_costs)
        discount_term = np.power(discount_fac, i)
        total_cost = (cost + total_procurement_cost) * discount_term
        total_costs.append(total_cost)

        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(policy_action)
        tau_sum += sourcingEnv.current_state.state_tau
        
        # Print diagnostic
        if debug_flag:
            event_tracker.append([event, event_index, policy_action, next_state.get_nested_list_repr(), total_cost ])
            count_demand = sum(map(lambda x : x[0] == Event.DEMAND_ARRIVAL, event_tracker))
            count_supply_0 = sum(map(lambda x : x[0] == Event.SUPPLY_ARRIVAL and x[1] == 1, event_tracker))
            count_supply_1 = sum(map(lambda x : x[0] == Event.SUPPLY_ARRIVAL and x[1] == 2, event_tracker))
            count_on_0 = sum(map(lambda x : x[0] == Event.SUPPLIER_ON and x[1] == 3, event_tracker))
            count_on_1 = sum(map(lambda x : x[0] == Event.SUPPLIER_ON and x[1] == 4, event_tracker))
            count_off_0 = sum(map(lambda x : x[0] == Event.SUPPLIER_OFF and x[1] == 5, event_tracker))
            count_off_1 = sum(map(lambda x : x[0] == Event.SUPPLIER_OFF and x[1] == 6, event_tracker))

    avg_cost_per_period = np.sum(total_costs)/tau_sum if not norm_m_flag else np.sum(total_costs)/len(total_costs)
    
    return total_costs, avg_cost_per_period

def mc_with_policy(sourcingEnv, 
    start_state = False,
    policy_callback = ss_policy_fastest_supp_backlog, 
    nested_mc_iters = NESTED_MC_ITERS,
    use_tqdm = False,
    **kwargs):
    
    mc_avg_costs = []

    for i in tqdm(range(nested_mc_iters)) if use_tqdm else range(nested_mc_iters):
        if start_state != False:
            sourcingEnv.current_state = start_state

        start_time = time.time()
        _, avg_cost = mc_episode_with_policy(sourcingEnv, policy = policy_callback, **kwargs)
        mc_avg_costs.append(avg_cost)
        run_time = time.time() - start_time
        # if i % 100 == 0:
        #     print("time per 100 iter: " + str(run_time))
    
    return mc_avg_costs

def dummy_explore_policy(sourcingEnv, **kwargs):
    if sourcingEnv.current_state.s < -1:
        return np.array([1, 1])
    else:
        return np.array([0, 1])