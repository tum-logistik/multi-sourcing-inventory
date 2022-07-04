import numpy as np
from common.variables import *
from scipy.stats import poisson
from sim.sim_functions import *
import yaml

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
    total_order_amount = np.clip(ss_pol_suggest- np.sum(sourcingEnv.current_state.n_backorders), 0, big_s)
    
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

    assert sourcingEnv.tracking_flag, "Assertion: Tracking feature must be on for dual index policy"

    ord_vec = np.zeros(len(sourcingEnv.supplier_lead_times_vec))
    # print("Opt vec: " + str(ord_vec_opt))
    # Safety cap
    if sourcingEnv.current_state.s >= safety_factor_di:
        ord_vec_opt = ss_policy_fastest_supp_backlog(sourcingEnv)
    elif sourcingEnv.current_state.s < 0:
        ord_vec_opt = ss_policy_fastest_supp_backlog(sourcingEnv)
        # return ord_vec_opt
    # elif False:
    else:
        tmark_exp = sourcingEnv.get_time_mark(sourcingEnv.action_history_tuple, sourcingEnv.exp_ind)
        tmark_reg = sourcingEnv.get_time_mark(sourcingEnv.action_history_tuple, sourcingEnv.reg_ind)

        # Demand in the expedited range
        demand_exp_range = np.array([x for x in sourcingEnv.demand_history_tuple if x[0] > tmark_exp])
        cum_demand_exp_range = np.sum(demand_exp_range, axis=0)[1] if len(demand_exp_range) > 0 else 0

        # Overshoot: defined as number of regular orders placed between n-le and n-lr
        overshoot_range = np.array([x for x in sourcingEnv.action_history_tuple if tmark_reg < x[0] < tmark_exp])
        action_history_ov_range = np.array([x[1] for x in overshoot_range])

        reg_orders_ov_range = action_history_ov_range[:, sourcingEnv.reg_ind] if len(action_history_ov_range) > 0 else np.array([])
        cum_reg_orders_ov_range = np.sum(reg_orders_ov_range)

        exp_orders_ov_range = action_history_ov_range[:, sourcingEnv.exp_ind] if len(action_history_ov_range) > 0 else np.array([])
        cum_exp_orders_ov_range = np.sum(exp_orders_ov_range)

        # critical fractal
        cf = b_penalty / (b_penalty + h_cost)
        min_cost = np.Inf
        ze_opt = 0
        zr_opt = 0
        delta_opt = 0
        gap = np.Inf

        for del_cand in range(-delta_cand_range, delta_cand_range):
            
            adj_val = cum_demand_exp_range - del_cand + cum_reg_orders_ov_range
            g_del = poisson.cdf(adj_val, sourcingEnv.lambda_arrival) #### Math Check 1 - poiss
            perc_cand = np.min([cf, g_del]) / cf 
            # perc_cand = (1 - g_del)  / cf
            ze, gap = inv_poisson(perc_cand, sourcingEnv.lambda_arrival)

            # ze, gap = inv_poisson(cf, sourcingEnv.lambda_arrival)
            zr = np.clip(ze + del_cand, 0, np.Inf) 

            ord_vec[sourcingEnv.exp_ind] = np.clip(ze - cum_reg_orders_ov_range, 0, np.Inf)
            ord_vec[sourcingEnv.reg_ind] = np.clip(zr - cum_exp_orders_ov_range, 0, np.Inf)

            delta_cost = cost_calc_expected_di(sourcingEnv, ord_vec)
            if delta_cost < min_cost:
                min_cost = delta_cost
                ze_opt = ze
                zr_opt = zr
                delta_opt = del_cand
                ord_vec_opt = ord_vec
    
        # ord_vec_opt = np.array([0, big_s - sourcingEnv.current_state.s])

        ord_vec_opt[sourcingEnv.exp_ind] = np.clip(ze_opt - sourcingEnv.current_state.s, 0, np.Inf)
        # reg_cap = np.max([0, int(sourcingEnv.current_state.s - ord_vec[sourcingEnv.exp_ind])])

        ord_vec_opt[sourcingEnv.reg_ind] = np.clip(zr_opt - sourcingEnv.current_state.s, 0, np.Inf)

    return ord_vec_opt


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
    cf = (b - procurement_cost) / (b + h)
    opt_inventory = inv_poisson(cf, lambda_arrival = lambda_arrival)
    return opt_inventory

def single_source_orderupto_policy(sourcingEnv, **kwargs):
    supplier_index = 0 if "supplier_index" not in kwargs else kwargs["supplier_index"]
    procurement_cost_vec = PROCUREMENT_COST_VEC if "procurement_cost_vec" not in kwargs else kwargs["procurement_cost_vec"]
    procurement_cost = procurement_cost_vec[supplier_index]
    opt_inventory, _ = newsvendor_opt_order(procurement_cost)
    order_amount = np.clip(opt_inventory - sourcingEnv.current_state.s, 0, np.Inf)

    action_array = np.zeros(sourcingEnv.n_suppliers)
    action_array[supplier_index] = order_amount
    return action_array


# implement kiesmueller heuristic,


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