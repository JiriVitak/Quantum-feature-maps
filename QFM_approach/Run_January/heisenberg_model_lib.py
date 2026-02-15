from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Layout
from qiskit.circuit import ParameterVector   # <<< NOVO
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import RZZGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from itertools import combinations
from collections import Counter

import json

import numpy as np

SEED = 42
np.random.seed(SEED)

