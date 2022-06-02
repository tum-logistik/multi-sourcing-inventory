from environment.SourcingEnv import *
import time

# Functional tests, 2, 3, 5, 10, 100 suppliers 99 iterations
def combination_func_tests():
    for n_sup in [2, 3, 5, 10, 100]:
        sourcingEnv = SourcingEnv(
                lambda_arrival = np.random.uniform(0, 100),
                procurement_cost_vec = np.ones(n_sup) * np.random.uniform(0, 100),
                supplier_lead_times_vec = np.ones(n_sup) * np.random.uniform(0, 100),
                on_times = np.ones(n_sup) * np.random.uniform(0, 100), 
                off_times = np.ones(n_sup) * np.random.uniform(0, 100)
            )
        functional_test(sourcingEnv)
    print("##### Passed all functional tests! #####")

    return True

def functional_test(sourcingEnv, iters = 99):
    start_time = time.time()
    for i in range(iters):
        random_action = np.array([np.random.randint(0, 4) for x in range(sourcingEnv.n_suppliers)])

        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(random_action)
        sum_check =  np.sum(probs)
    
    print("##### Test Run with (" + str(sourcingEnv.n_suppliers) + ") suppliers passed! #####")

    run_time = time.time() - start_time
    print("Time -func test- no. suppliers " + str(sourcingEnv.n_suppliers) + ") - runtime: " + str(run_time))
    return True, run_time

def assert_state_equality(next_state, check_state, ass_msg = "Test Failed: "):
    assert next_state.s == check_state.s and (next_state.flag_on_off == check_state.flag_on_off).all() and (next_state.n_backorders == check_state.n_backorders).all(), ass_msg

# Case tests - Run on 2 agents [0, 0, 0, 0, 1, 1]
def run_test_scenario_1(sourcingEnv, n_sup = 2):
    # Test 1
    # DEMAND_ARRIVAL (2, 1) -> DEMAND_ARRIVAL (3, 1) -> SUPPLY_ARRIVAL_1 -> (3, 2)
    # Backorder: [-1, 2, 1, 1, 1] -> [-2, 5, 2, 1, 1] -> [0, 8, 2, 1, 1]
    # LS:
    
    start_state = sourcingEnv.reset()
    scenario_id = "1"
    
    next_state, _, _, _, _ = sourcingEnv.step(np.array([2, 1]), force_event_tuple = (Event.DEMAND_ARRIVAL, None))
    check_state = MState(-1, n_sup, n_backorders = np.array([2, 1]), flag_on_off = np.array([1, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([3, 1]), force_event_tuple = (Event.DEMAND_ARRIVAL, None))
    check_state = MState(-2, n_sup, n_backorders = np.array([5, 2]), flag_on_off = np.array([1, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([3, 2]), force_event_tuple = (Event.SUPPLY_ARRIVAL, 1))
    check_state = MState(0, n_sup, n_backorders = np.array([8, 2]), flag_on_off = np.array([1, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    print("##### Test Scenario (" + scenario_id + ") passed #####")

    return True

def run_test_scenario_2(sourcingEnv, n_sup = 2):
    # Test 2
    # SUPPLY_ARRIVAL_0 (5, 5) -> SUPPLIER_OFF_0 (200, 3) -> SUPPLY_ARRIVAL_0 (5, 5)  -> SUPPLIER_ARRIVAL_1 (9, 2)
    # [0, 5, 5, 1, 1] -> [0, 205, 8, 0, 1] -> [205, 0, 13, 0, 1] -> [218, 0, 2, 0, 1]

    start_state = sourcingEnv.reset()
    scenario_id = "2"
    
    next_state, _, _, _, _ = sourcingEnv.step(np.array([5, 5]), force_event_tuple = (Event.SUPPLY_ARRIVAL, 0))
    check_state = MState(0, n_sup, n_backorders = np.array([5, 5]), flag_on_off = np.array([1, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([200, 3]), force_event_tuple = (Event.SUPPLIER_OFF, 0))
    check_state = MState(0, n_sup, n_backorders = np.array([205, 8]), flag_on_off = np.array([0, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([5, 5]), force_event_tuple = (Event.SUPPLY_ARRIVAL, 0))
    check_state = MState(205, n_sup, n_backorders = np.array([0, 13]), flag_on_off = np.array([0, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([9, 2]), force_event_tuple = (Event.SUPPLY_ARRIVAL, 1))
    check_state = MState(218, n_sup, n_backorders = np.array([0, 2]), flag_on_off = np.array([0, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    print("##### Test Scenario (" + scenario_id + ") passed #####")

    return True

def run_test_scenario_3(sourcingEnv, n_sup = 2):
    start_state = sourcingEnv.reset()
    scenario_id = "3"

    next_state, _, _, _, _ = sourcingEnv.step(np.array([5, 4]), force_event_tuple = (Event.SUPPLIER_OFF, 0))
    check_state = MState(0, n_sup, n_backorders = np.array([5, 4]), flag_on_off = np.array([0, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([5, 5]), force_event_tuple = (Event.SUPPLIER_OFF, 0))
    check_state = MState(0, n_sup, n_backorders = np.array([5, 9]), flag_on_off = np.array([0, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([3, 3]), force_event_tuple = (Event.SUPPLIER_ON, 0))
    check_state = MState(0, n_sup, n_backorders = np.array([5, 12]), flag_on_off = np.array([1, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([2, 1]), force_event_tuple = (Event.SUPPLIER_ON, 0))
    check_state = MState(0, n_sup, n_backorders = np.array([7, 13]), flag_on_off = np.array([1, 1]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([7, 6]), force_event_tuple = (Event.SUPPLIER_OFF, 1))
    check_state = MState(0, n_sup, n_backorders = np.array([14, 19]), flag_on_off = np.array([1, 0]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    next_state, _, _, _, _ = sourcingEnv.step(np.array([10, 10]), force_event_tuple = (Event.SUPPLY_ARRIVAL, 1))
    check_state = MState(19, n_sup, n_backorders = np.array([24, 0]), flag_on_off = np.array([1, 0]))
    assert_state_equality(next_state, check_state, ass_msg = "Test Failed: Scenario " + scenario_id)

    print("##### Test Scenario (" + scenario_id + ") passed #####")
    return True
    
        
