# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CNOT test
"""

from paddle import to_tensor, matmul
from paddle_quantum.mbqc.utils import pauli_gate, cnot_gate, basis, random_state_vector
from paddle_quantum.mbqc.utils import compare_by_vector, compare_by_density
from paddle_quantum.mbqc.simulator import MBQC
from paddle_quantum.mbqc.qobject import State

X = pauli_gate('X')
Z = pauli_gate('Z')
X_basis = basis('X')
Y_basis = basis('Y')

# Construct the underlying graph of CNOT implementation in MBQC
V = [str(i) for i in range(1, 16)]
E = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),
     ('5', '6'), ('6', '7'), ('4', '8'), ('8', '12'),
     ('9', '10'), ('10', '11'), ('11', '12'),
     ('12', '13'), ('13', '14'), ('14', '15')]
G = [V, E]

# Generate a random state vector
input_psi = random_state_vector(2, is_real=False)

# Instantiate a MBQC class
mbqc = MBQC()
# Set the underlying graph for computation
mbqc.set_graph(G)
# Set the input state
mbqc.set_input_state(State(input_psi, ['1', '9']))
# Watch the computational process
# mbqc.draw_process()

# Start measurement process
mbqc.measure('1', X_basis)
mbqc.measure('2', Y_basis)
mbqc.measure('3', Y_basis)
mbqc.measure('4', Y_basis)
mbqc.measure('5', Y_basis)
mbqc.measure('6', Y_basis)
mbqc.measure('8', Y_basis)
mbqc.measure('9', X_basis)
mbqc.measure('10', X_basis)
mbqc.measure('11', X_basis)
mbqc.measure('12', Y_basis)
mbqc.measure('13', X_basis)
mbqc.measure('14', X_basis)

# Obtain byproduct's exponents
cx = mbqc.sum_outcomes(['2', '3', '5', '6'])
tx = mbqc.sum_outcomes(['2', '3', '8', '10', '12', '14'])
cz = mbqc.sum_outcomes(['1', '3', '4', '5', '8', '9', '11'], 1)
tz = mbqc.sum_outcomes(['9', '11', '13'])

# Correct byproducts
mbqc.correct_byproduct('X', '7', cx)
mbqc.correct_byproduct('X', '15', tx)
mbqc.correct_byproduct('Z', '7', cz)
mbqc.correct_byproduct('Z', '15', tz)

# Obtain the output state
state_out = mbqc.get_quantum_output()

# Find the standard result
vec_std = matmul(to_tensor(cnot_gate()), input_psi)
system_std = ['7', '15']
state_std = State(vec_std, system_std)
# Compare with the standard result
compare_by_vector(state_out, state_std)
compare_by_density(state_out, state_std)
