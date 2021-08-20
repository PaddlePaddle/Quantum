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

from paddle_quantum.mbqc.utils import random_state_vector, compare_by_vector, compare_by_density
from paddle_quantum.mbqc.simulator import MBQC
from paddle_quantum.mbqc.qobject import Circuit, State
from paddle_quantum.mbqc.mcalculus import MCalculus
from paddle_quantum.circuit import UAnsatz

n = 2  # Set the circuit width
# Generate a random state vector
input_psi = random_state_vector(2, is_real=False)

# Instantiate a Circuit class
cir = Circuit(n)

# There are two patterns for CNOT gate, one is the 4-nodes pattern, the other is the 15-nodes pattern
# We can use CNOT gate in definite pattern that we want
# cir.cnot([0,1])  # CNOT pattern with 4 - nodes
cir.cnot_15([0, 1])  # CNOT pattern with 15 - nodes

# Instantiate a MCalculus class
mc = MCalculus()
mc.set_circuit(cir)
# If we want to do pattern standardization, signal shifting, measurement order optimization,
# we can use the following lines in order. However, as one CNOT gate is already a standardized pattern itself,
# there is no need to do so.
# mc.standardize()
# mc.shift_signals()
# mc.optimize_by_row()
pattern = mc.get_pattern()

# Instantiate a MBQC class
mbqc = MBQC()
mbqc.set_pattern(pattern)
# mbqc.draw_process(draw=True)
mbqc.set_input_state(State(input_psi, [0, 1]))
# Run computation by pattern
mbqc.run_pattern()
# Obtain the output state
state_out = mbqc.get_quantum_output()

# Find the standard result
cir_std = UAnsatz(n)
cir_std.cnot([0, 1])
vec_std = cir_std.run_state_vector(input_psi.astype("complex128"))
system_std = state_out.system
state_std = State(vec_std, system_std)
# Compare with the standard result in UAnsatz
compare_by_vector(state_out, state_std)
compare_by_density(state_out, state_std)
