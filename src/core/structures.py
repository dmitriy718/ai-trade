"""
Core Data Structures - Optimized data containers for high-performance trading.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class RingBuffer:
    """
    Fixed-size circular buffer using pre-allocated NumPy arrays.
    Optimized for O(1) appends and zero-copy sliding windows.
    """

    def __init__(self, capacity: int, dtype: np.dtype = np.float64):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self._data = np.zeros(capacity, dtype=dtype)

    def append(self, value: float) -> None:
        """Append a single value to the buffer, overwriting oldest if full."""
        self._data[self.position] = value
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def append_many(self, values: np.ndarray) -> None:
        """Append multiple values efficiently."""
        n = len(values)
        if n >= self.capacity:
            # Overwrite everything
            self._data[:] = values[-self.capacity:]
            self.position = 0
            self.size = self.capacity
            return

        # Calculate indices
        start = self.position
        end = (start + n) % self.capacity

        if start + n <= self.capacity:
            # No wrap-around
            self._data[start:start+n] = values
        else:
            # Wrap-around
            split = self.capacity - start
            self._data[start:] = values[:split]
            self._data[:n-split] = values[split:]

        self.position = end
        if self.size < self.capacity:
            self.size = min(self.size + n, self.capacity)

    def view(self) -> np.ndarray:
        """
        Return a view of the valid data in chronological order.
        Note: This may return a copy if the data wraps around.
        """
        if self.size < self.capacity:
            return self._data[:self.size]
        
        # If full, we need to unroll
        return np.concatenate((
            self._data[self.position:], 
            self._data[:self.position]
        ))

    def latest(self, n: int = 1) -> np.ndarray:
        """Get the most recent n values."""
        if n <= 0:
            return np.array([])
        n = min(n, self.size)
        
        end = self.position
        start = (end - n) % self.capacity
        
        if start < end:
            return self._data[start:end]
        else:
            return np.concatenate((self._data[start:], self._data[:end]))

    def get_last(self) -> float:
        """Get the very last appended value."""
        if self.size == 0:
            return 0.0
        idx = (self.position - 1) % self.capacity
        return self._data[idx]

    def clear(self) -> None:
        """Reset the buffer."""
        self.size = 0
        self.position = 0
        self._data.fill(0)
