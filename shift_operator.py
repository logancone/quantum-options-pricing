from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

num_qubits = 6
qc = QuantumCircuit(num_qubits)

# Adds controlled-x gates iteratively
for i in range(num_qubits-1, 0, -1):
    qc.mcx(list(range(0,i)), i)
    
# Final flip gate
qc.x(0)
    
print(qc.draw())

q_state = Statevector.from_instruction(qc).data
print(q_state)