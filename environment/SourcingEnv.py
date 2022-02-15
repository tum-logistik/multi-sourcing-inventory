import numpy as np
from DualSourcingEvent import *

class SourcingEnv():

    # Switch to np.array for speed up
    def __init__(self, order_quantity = 30, 
        procurement_cost_vec = np.array([2, 1.7]), 
        supplier_lead_times_vec = np.array([0.5, 0.75]), 
        on_times = np.array([3, 1]), 
        off_times = np.array([0.3, 1])):
        
        invert_np = lambda x: 1/x

        self.action_size = order_quantity
        self.lambda_arrival = 0.1
        self.on_times = on_times
        self.off_times = off_times
        self.procurement_cost_vec = procurement_cost_vec
        self.supplier_lead_times_vec = supplier_lead_times_vec
        self.n_suppliers = len(procurement_cost_vec)
        self.current_state = self.reset()

        self.mu_lead_time = invert_np(self.supplier_lead_times_vec)
        self.mu_on_times = invert_np(self.on_times)
        self.mu_off_times = invert_np(self.off_times)  

    # order_quantity_vec is action
    def compute_event_rate(self, order_quantity_vec, 
        state_vec = None, 
        lambda_arrival = None, 
        mu_lead_time = None, 
        mu_on_times = None, 
        mu_off_times = None):
        
        if lambda_arrival == None:
            lambda_arrival = self.lambda_arrival
        if mu_lead_time == None:
            mu_lead_time = self.mu_lead_time
        if mu_on_times == None:
            mu_on_times = self.mu_on_times
        if mu_off_times == None:
            mu_off_times = self.mu_off_times
        if state_vec == None:
            state_vec = self.current_state
        
        outstd_orders = state_vec[1:1+self.n_suppliers]
        onoff_status = state_vec[1+self.n_suppliers:]

        lead_time_comps = np.multiply(order_quantity_vec + outstd_orders, mu_lead_time)
        lead_time_comp = np.sum(lead_time_comps)
        mu_off_comp = np.sum(np.multiply(1 - onoff_status, mu_off_times))
        mu_on_comp = np.sum(np.multiply(1 - onoff_status, mu_on_times))

        event_rate = lambda_arrival + lead_time_comp + mu_off_comp + mu_on_comp

        return 1 / event_rate
    
    # index for each event
    # k is supplier index
    def compute_trans_prob(self, order_quantity_vec, event_type, k = None, state_vec = None):

        if state_vec == None:
            state_vec = self.current_state
        
        if event_type == DualSourcingEvent.DEMAND_ARRIVAL:
            return self.lambda_arrival * self.compute_event_rate(order_quantity_vec)
        else:
            assert k is not None
            if event_type == DualSourcingEvent.SUPPLY_ARRIVAL:
                outstd_orders = self.state_vec[1+k]
                return (outstd_orders + order_quantity_vec[k]) * self.mu_lead_time[k] * self.compute_event_rate(order_quantity_vec)

            if event_type == DualSourcingEvent.SUPPLIER_ON:
                onoff_status = state_vec[1+self.n_suppliers+k]
                return (1 - onoff_status) * self.mu_off_times[k] * self.compute_event_rate(order_quantity_vec)
            
            if event_type == DualSourcingEvent.SUPPLIER_OFF:
                onoff_status = state_vec[1+self.n_suppliers+k]
                return onoff_status * self.mu_on_times[k] * self.compute_event_rate(order_quantity_vec)
        
        return 0



    # state is defined as inventories of each agent + 
    def reset(self):
        return np.array([0] + [0] * self.n_suppliers + [1] * self.n_suppliers)
    
    # .step(action)
    def step(self):
        return 1
    