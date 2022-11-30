from enum import Enum
import numpy as np
from common.variables import *

# indexed 0,1,2,3...
class Event(Enum):
    DEMAND_ARRIVAL = 0
    SUPPLY_ARRIVAL = 1
    SUPPLIER_ON = 2
    SUPPLIER_OFF = 3
    NO_EVENT = 4

# indexed 0,1,2,3...
class MState():

    def __init__(self,
        stock_level = 0, 
        n_suppliers = N_SUPPLIERS,
        n_backorders = False,
        flag_on_off = False,
        state_tau = 0.0):
        
        self.s = stock_level # stock level
        self.n_backorders = n_backorders if isinstance(n_backorders, np.ndarray) else np.zeros(n_suppliers)
        self.flag_on_off = flag_on_off if isinstance(flag_on_off, np.ndarray) else np.ones(n_suppliers)
        self.state_tau = state_tau
        # self.flag_on_off = np.ones(n_suppliers) if not flag_on_off.any() else flag_on_off # on off flag
    
    def __str__(self):
        return "Stock: {fname}, n backorders: {nb}, supplier status (on/off): {sup_stat}".format(fname = self.s, nb = self.n_backorders, sup_stat = self.flag_on_off)

    def get_list_repr(self):
        arr_rep = [int(self.s)] + [int(x) for x in self.n_backorders] + [int(x) for x in self.flag_on_off]
        return arr_rep

    def get_nested_list(self):
        arr_rep = [int(self.s), [int(x) for x in self.n_backorders], [int(x) for x in self.flag_on_off]]
        return arr_rep

    def get_nested_list_repr(self):
        return repr(self.get_nested_list())

    def get_repr_key(self):
        return repr(self.get_list_repr())
    

