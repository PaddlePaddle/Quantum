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
Single qubit unitary gate test with random angle
"""

from numpy import pi, random
from paddle import to_tensor, matmul
from paddle_quantum.mbqc.utils import random_state_vector, rotation_gate, compare_by_vector
from paddle_quantum.mbqc.simulator import MBQC
from paddle_quantum.mbqc.qobject import Circuit, State
from paddle_quantum.mbqc.mcalculus import MCalculus

# Set qubit number
n = 1

# Input vector and label
# If we want to choose an input state randomly, we can use ``random_state_vector``
input_psi = random_state_vector(1, is_real=False)

# To be simplify, here we choose "|+>" as the input vector
# input_vec = plus_state()

# Default that 'alpha' represents the initial rotation gates' angles without adaptive constants
# Default that 'theta' represents the adaptive measurement angles

# Set 'alpha'
alpha = to_tensor([2 * pi * random.uniform()], dtype='float64')
beta = to_tensor([2 * pi * random.uniform()], dtype='float64')
gamma = to_tensor([2 * pi * random.uniform()], dtype='float64')

# Note: Here the parameters are not equal to those in UAnsatz circuit's "U3".
# Indeed, we decompose unitary matrix as U = Rz Rx Rz instead of U = Rz Ry Rz in UAnsatz
params = [alpha, beta, gamma]

# Initialize circuit
cir = Circuit(n)
cir.u(params, 0)
# circuit = cir.get_circuit()

# Initialize pattern
pat = MCalculus()
pat.set_circuit(cir)

# If we want to standardize the circuit, shift signals or optimize the circuit, we can use the following lines
# Note: As one CNOT gate has already been a standardized pattern, there is no need to do so.
# pat.standardize()
# pat.shift_signals()
# pat.optimize_by_row()
pattern = pat.get_pattern()

# Initialize MBQC
mbqc = MBQC()
mbqc.set_pattern(pattern)
mbqc.set_input_state(State(input_psi, [0]))
# If we want to plot the process of measurement, we can call th function ``mbqc.plot()``
# mbqc.plot(pause_time=1.0)
# Run by pattern
mbqc.run_pattern()
# Acquire the output state
state_out = mbqc.get_quantum_output()

# Compare with the standard result
vec_std = matmul(rotation_gate('z', gamma),
                 matmul(rotation_gate('x', beta),
                        matmul(rotation_gate('z', alpha), input_psi)))
system_label = state_out.system
state_std = State(vec_std, system_label)
compare_by_vector(state_out, state_std)
