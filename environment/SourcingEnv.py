import numpy as np

class SourcingEnv():

    # Switch to np.array for speed up
    def __init__(self, order_quantity = 30, n_sources = 2, supplier_costs = [2, 1.7], supplier_lead_times = [0.5, 0.75], on_times = [3, 1], off_times = [0.3, 1]):
        self.action_size = order_quantity
        self.lambda_arrival = 0.1
        self.lead_times = supplier_lead_times
        self.on_times = on_times
        self.off_times = off_times
    

    def compute_event_rate(self, order_quantity_vec, supplier_state_vec, lambda_arrival = None, mu_lead_time = None, mu_on_times = None, mu_off_times = None):
        
        if lambda_arrival == None:
            lambda_arrival = self.lambda_arrival
        if mu_lead_time == None:
            mu_lead_time = [1/x for x in self.lead_times]
        if mu_on_times == None:
            mu_on_times = [1/x for x in self.on_times]
        if mu_off_times == None:
            mu_off_times = [1/x for x in self.off_times]

        lead_time_component = 1
        mu_off_comp = 1
        mu_on_comp = 1



        return 1


    
    # state is defined as inventories of each agent + 
    def reset(self):
        return 1
    
    # .step(action)
    def step(self):
        return 1
    