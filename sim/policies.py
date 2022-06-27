import numpy as np
from common.variables import *

def ss_policy(inventory_level, small_s, big_s):

    assert big_s > small_s, "Assertion Failed: Big S smaller than small S"

    if inventory_level < small_s:
        return big_s - inventory_level
    else:
        return 0

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


# order from quickest available supplier
def ss_policy_fastest_supp_backlog(sourcingEnv, small_s = SMALL_S, big_s = BIG_S):

    ss_pol_suggest = ss_policy(sourcingEnv.current_state.s, small_s = small_s, big_s = big_s) 
    total_order_amount = np.clip(ss_pol_suggest- np.sum(sourcingEnv.current_state.n_backorders), 0, big_s)
    
    avail_mu_lt_prod = sourcingEnv.mu_lt_rate * sourcingEnv.current_state.flag_on_off
    supp_index = np.argmax(avail_mu_lt_prod)

    policy_action = np.zeros(sourcingEnv.n_suppliers)
    policy_action[supp_index] = total_order_amount
    
    return policy_action


# implement dual index policy
def dual_index_policy(sourcingEnv):
    assert sourcingEnv.tracking_flag, "Assertion: Tracking feature must be on for dual index policy"

    tmark_exp = sourcingEnv.get_time_mark(sourcingEnv.action_history_tuple, sourcingEnv.exp_ind)
    tmark_reg = sourcingEnv.get_time_mark(sourcingEnv.action_history_tuple, sourcingEnv.reg_ind)

    # Demand in the expedited range
    # TODO: Create time slicer filter
    demand_exp_range = np.array([x for x in sourcingEnv.demand_history_tuple if x[0] > tmark_exp])

    cum_demand_exp_range = np.sum(demand_exp_range, axis=0)[1]

    # Overshoot: defined as number of regular orders placed between n-le and n-lr
    overshoot_range = np.array([x for x in sourcingEnv.action_history_tuple if tmark_reg < x[0] < tmark_exp])

    action_history_ov_range = np.array([x[1] for x in overshoot_range])

    reg_orders_ov_range = action_history_ov_range[:, sourcingEnv.reg_ind] if len(action_history_ov_range) > 0 else np.array([])
    cum_reg_orders_ov_range = np.sum(reg_orders_ov_range)

    return tmark_exp, tmark_reg


# implement single supplier newsvendor,
# implement kiesmueller heuristic,
