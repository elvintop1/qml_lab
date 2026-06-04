from abc import ABC, abstractmethod
import numpy as np
from qiskit import QuantumCircuit
class Encoding(ABC):
    def __init__(self, num_qubits: int, **kwargs):
        self.num_qubits = num_qubits
        self.params = dict(kwargs)
    @abstractmethod
    def build(self, x: np.ndarray) -> QuantumCircuit:
        raise NotImplementedError
    def __repr__(self):
        cname = getattr(self, "encoding_name", self.__class__.__name__)
        return f"{cname}(num_qubits={self.num_qubits}, params={self.params})"
