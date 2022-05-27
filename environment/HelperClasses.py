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
        n_suppliers = 2):
        
        self.s = stock_level # stock level
        self.n_backorders = np.zeros(n_suppliers) # n outstanding backorders
        self.flag_on_off = np.ones(n_suppliers) # on off flag




