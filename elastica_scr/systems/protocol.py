__doc__ = """Base class for SCR system"""

from typing import Protocol, Type, runtime_checkable

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from elastica.systems.protocol import StaticSystemProtocol


@runtime_checkable
class ImplicitSystemProtocol(StaticSystemProtocol, Protocol):
    """
    Protocol for all dynamic elastica system.
    """

    @abstractmethod
    def update_orientation(self, time: np.float64) -> None: ...

    @abstractmethod
    def update_position(self, time: np.float64) -> None: ...

    @abstractmethod
    def update_velocity(self, time: np.float64) -> None: ...

    @abstractmethod
    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None: ...
