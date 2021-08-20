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
from paddle_quantum.mbqc.utils import random_state_vector, rotation_gate, basis
from paddle_quantum.mbqc.utils import compare_by_vector, compare_by_density
from paddle_quantum.mbqc.simulator import MBQC
from paddle_quantum.mbqc.qobject import State

# Construct the underlying graph of single-qubit implementation in MBQC
G = [['1', '2', '3', '4', '5'], [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]]

# Generate a random state vector
input_vec = random_state_vector(1, is_real=False)

# Suppose the single-qubit gate is decomposed by Rx(alpha_4) Rz(alpha_3) Rx(alpha_2)
alpha_1 = 0
alpha_2 = pi * random.uniform()
alpha_3 = pi * random.uniform()
alpha_4 = pi * random.uniform()

# Instantiate a MBQC class
mbqc = MBQC()
# Set the underlying graph for computation
mbqc.set_graph(G)
# Set the input state
mbqc.set_input_state(State(input_vec, ['1']))
# Watch the computational process
# mbqc.draw_process(pause_time=1.0)

# Set the measurement angles
# Measure qubit '1', with "theta = alpha"
theta_1 = alpha_1
theta_1 = to_tensor([theta_1], dtype='float64')
mbqc.measure('1', basis('XY', theta_1))

# Measure qubit '2', with "theta = (-1)^{s_1 + 1} * alpha"
theta_2 = (-1) ** mbqc.sum_outcomes(['1'], 1) * alpha_2
theta_2 = to_tensor([theta_2], dtype='float64')
mbqc.measure('2', basis('XY', theta_2))

# Measure qubit '3', with "theta = (-1)^{s_2 + 1} * alpha"
theta_3 = (-1) ** mbqc.sum_outcomes(['2'], 1) * alpha_3
theta_3 = to_tensor([theta_3], dtype='float64')
mbqc.measure('3', basis('XY', theta_3))

# Measure qubit '4', with "theta = (-1)^{s_1 + s_3 + 1} * alpha"
theta_4 = (-1) ** mbqc.sum_outcomes(['1', '3'], 1) * alpha_4
theta_4 = to_tensor([theta_4], dtype='float64')
mbqc.measure('4', basis('XY', theta_4))

# Correct byproduct operators
mbqc.correct_byproduct('X', '5', mbqc.sum_outcomes(['2', '4']))
mbqc.correct_byproduct('Z', '5', mbqc.sum_outcomes(['1', '3']))

# Obtain the output state
state_out = mbqc.get_quantum_output()

# Find the standard result
vec_std = matmul(rotation_gate('x', to_tensor([alpha_4], dtype='float64')),
                 matmul(rotation_gate('z', to_tensor([alpha_3], dtype='float64')),
                        matmul(rotation_gate('x', to_tensor([alpha_2], dtype='float64')), input_vec)))
system_std = ['5']
state_std = State(vec_std, system_std)
# Compare with the standard result
compare_by_vector(state_out, state_std)
compare_by_density(state_out, state_std)
