from typing import Optional
from collections import namedtuple


MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats:
    """A class that holds the min-max values of the tree.

    Ref: MuZero pseudocode
    """

    def __init__(self, known_bounds: Optional[KnownBounds]):
        if known_bounds:
            self.maximum = known_bounds.max
            self.minimum = known_bounds.min
        else:
            self.maximum = -MAXIMUM_FLOAT_VALUE
            self.minimum = MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        """Update min and mad values."""
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """Normalize value given known min and max values."""
        if self.maximum > self.minimum:
            # Normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    def __str__(self):
        return f"MinMaxState: (minimum: {self.minimum}, maximum: {self.maximum})"
