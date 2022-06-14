import numpy as np

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
def ss_policy_fastest_supp_backlog(sourcingEnv, small_s, big_s):

    ss_pol_suggest = ss_policy(sourcingEnv.current_state.s, small_s = small_s, big_s = big_s) 
    total_order_amount = np.clip(ss_pol_suggest- np.sum(sourcingEnv.current_state.n_backorders), 0, big_s)
    
    avail_mu_lt_prod = sourcingEnv.mu_lt_rate * sourcingEnv.current_state.flag_on_off
    supp_index = np.argmax(avail_mu_lt_prod)

    policy_action = np.zeros(sourcingEnv.n_suppliers)
    policy_action[supp_index] = total_order_amount
    
    return policy_action
# implement single supplier newsvendor,
# implement kiesmueller heuristic,

