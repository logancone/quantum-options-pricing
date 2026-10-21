import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

qubit_count = 6

qc = QuantumCircuit(qubit_count)

# qc.h(0)
# qc.cx(0, 1)
for i in range(qubit_count):
    qc.h(i)

qc.ccx(0, 2, 4)
qc.cswap(1, 3, 5)
qc.swap(3, 4)
qc.cx(3, 2)
qc.s(0)

q_state = Statevector.from_instruction(qc).data

# print(Operator(qc).data)
# np_matrix = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0], [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)], [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2), 0, 0]])

np_matrix = np.array(Operator(qc).data)

initial_state = np.zeros(2**qubit_count, dtype=complex)
initial_state[0] = 1

np_state = np_matrix @ initial_state

# print(f"Quantum State: {q_state}")
# print(f"NP State: {np_state}")
print(f"Unitary?: {np.allclose(np_matrix@np_matrix.conj().T,np.identity(2**qubit_count))}")
print(f"Do they match?: {np.allclose(q_state, np_state)}")

