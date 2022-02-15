from enum import Enum

# indexed 0,1,2,3...
class DualSourcingEvent(Enum):
    DEMAND_ARRIVAL = 0
    SUPPLY_ARRIVAL = 1
    SUPPLIER_ON = 2
    SUPPLIER_OFF = 3

