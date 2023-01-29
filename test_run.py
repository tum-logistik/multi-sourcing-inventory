from tests.base_tests import *

def test_sourcing_env_smdp():
    combination_func_tests()

    sourcingEnv = SourcingEnv(
        lambda_arrival = 5, # or 10
        procurement_cost_vec = np.array([3, 1]),
        supplier_lead_times_vec = np.array([0.5, 0.75]),
        on_times = np.array([3, 1]), 
        off_times = np.array([0.3, 1]))
    
    run_test_scenario_1(sourcingEnv)
    run_test_scenario_2(sourcingEnv)
    run_test_scenario_3(sourcingEnv)

if __name__ == '__main__':
    
    print("Test MDP")

    premium_thresh = -3
    small_s = 3
    big_s = 6
    t_horizon = 30

    sourcingEnv = SourcingEnvMDP(
        lambda_arrival = 2, # or 10
        procurement_cost_vec = np.array([3, 1]),
        supplier_lead_times_vec = np.array([0.5, 0.75]),
        on_times = np.array([3, 1]), 
        off_times = np.array([0.3, 1]))
    
    cum_cost = 0.0

    print(sourcingEnv.current_state)
    for i in range(t_horizon):
        order_amt = np.array([0, 0])
        if sourcingEnv.current_state.s < small_s:
            if sourcingEnv.current_state.s < premium_thresh:
                order_amt[0] = big_s - sourcingEnv.current_state.s
            else:
                order_amt[1] = big_s - sourcingEnv.current_state.s
        next_state, demand_arrivals, order_arrivals, flag_on_off  = sourcingEnv.step(order_amt)
        cost = cost_calc_unit(next_state)
        cum_cost += cost
        print(demand_arrivals, order_arrivals, cum_cost/i)
        print(sourcingEnv.current_state)
    
