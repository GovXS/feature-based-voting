from enum import Enum

class ElicitationMethod(Enum):
    FRACTIONAL = "fractional"
    CUMULATIVE = "cumulative"
    APPROVAL = "approval"
    PLURALITY = "plurality"

class AggregationMethod(Enum):
    ARITHMETIC_MEAN = "arithmetic_mean"
    MEDIAN = "median" 