from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit  


@dataclass
class RunResult:
    y_true: np.ndarray
    y_pred: np.ndarray

    y_prob: Optional[np.ndarray] = None  

    K_train: Optional[np.ndarray] = None
    K_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None  

    circuits: Optional[List["QuantumCircuit"]] = None

    model_name: Optional[str] = None
    encoding_name: Optional[str] = None
    encoding_params: Optional[Dict[str, Any]] = None
    feat_idx: Optional[List[int]] = None

    num_params: Optional[int] = None       
    shots: Optional[int] = None           
    evals: Optional[int] = None            
    timings: Dict[str, float] = field(default_factory=dict)  

    extras: Dict[str, Any] = field(default_factory=dict)
