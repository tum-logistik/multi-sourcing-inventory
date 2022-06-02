from env.SourcingEnv import *
from sim.sim_functions import *
from sim.policies import *
import numpy as np

if __name__ == '__main__':
    print("#### Running Main Simulation ####")

    sourcingEnv = SourcingEnv(
        lambda_arrival = 599, # or 10
        procurement_cost_vec = np.array([3, 1, 2]),
        supplier_lead_times_vec = np.array([0.5, 0.75, 1.0]),
        on_times = np.array([3, 1, 2]), 
        off_times = np.array([0.3, 1, 1.5]))
    
    history = []

    cost = cost_calc(sourcingEnv.current_state, h_cost = 4, b_penalty = 6)
    print(str(sourcingEnv.current_state) + " cost: " + str(cost))

    for i in range(300):
        
        # totally random order amounts
        # random_action = np.array([np.random.randint(0, 2) for x in range(sourcingEnv.n_suppliers)])

        # s S policy with aribtrary s S, and randomly selected vendor
        policy_action = ss_policy_with_randomly_select_vendor(sourcingEnv, small_s = 3, big_s = 15)
        
        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(policy_action)

        cost = cost_calc(next_state, h_cost = 4, b_penalty = 6)
        print(str(next_state) + " cost: " + str(cost) + " event: " + str(event) + " action:" + str(policy_action))
        
        history.append([next_state, event, i, probs])
    
