from enum import Enum
import numpy as np

# indexed 0,1,2,3...
class Event(Enum):
    DEMAND_ARRIVAL = 0
    SUPPLY_ARRIVAL = 1
    SUPPLIER_ON = 2
    SUPPLIER_OFF = 3
    NO_EVENT = 4

# indexed 0,1,2,3...
# indexed 0,1,2,3...
class MState():

    def __init__(self,
        stock_level = 0, 
        n_suppliers = 2,
        n_backorders = False,
        flag_on_off = False):
        
        self.s = stock_level # stock level
        self.n_backorders = n_backorders if isinstance(n_backorders, np.ndarray) else np.zeros(n_suppliers)
        self.flag_on_off = flag_on_off if isinstance(flag_on_off, np.ndarray) else np.ones(n_suppliers)
        # self.flag_on_off = np.ones(n_suppliers) if not flag_on_off.any() else flag_on_off # on off flag
    
    def __str__(self):
        return "Stock: {fname}, n backorders: {nb}, supplier status (on/off): {sup_stat}".format(fname = self.s, nb = self.n_backorders, sup_stat = self.flag_on_off)

