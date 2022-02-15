from enum import Enum

# indexed 0,1,2,3...
class Event(Enum):
    DEMAND_ARRIVAL = 0
    SUPPLY_ARRIVAL = 1
    SUPPLIER_ON = 2
    SUPPLIER_OFF = 3
    NO_EVENT = 4

