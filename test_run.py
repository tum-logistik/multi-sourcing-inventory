from tests.base_tests import *

if __name__ == '__main__':

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
