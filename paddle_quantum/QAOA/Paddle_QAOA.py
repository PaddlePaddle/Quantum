# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
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
Paddle_QAOA: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import os
from paddle import fluid
from paddle.complex import matmul as pp_matmul
from paddle.complex import transpose
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.QAOA.QAOA_Prefunc import generate_graph, H_generator

from numpy import ones, abs, conjugate, real, savez, sqrt, zeros
from numpy import matmul as np_matmul
from numpy import pi as PI

# Random seed for optimizer
SEED = 1

__all__ = [
    "circuit_QAOA",
    "circuit_extend_QAOA",
    "Net",
    "Paddle_QAOA",
]


def circuit_QAOA(theta, input_state, adjacency_matrix, N, P):
    """
    This function constructs the parameterized QAOA circuit which is composed of P layers of two blocks:
    one block is U_theta[layer][0] based on the problem Hamiltonian H which encodes the classical problem,
    and the other is U_theta[layer][1] constructed from the driving Hamiltonian describing the rotation around Pauli X
    acting on each qubit. It finally outputs the final state of the QAOA circuit.

    Args:
         theta: parameters to be optimized in the QAOA circuit
         input_state: initial state of the QAOA circuit which usually is the uniform superposition of 2^N bit-strings
                    in the computational basis $|0\rangle, |1\rangle$
         adjacency_matrix:  the adjacency matrix of the graph encoding the classical problem
         N: number of qubits, or equivalently, the number of nodes in the given graph
         P: number of layers of two blocks in the QAOA circuit
    Returns:
        the final state of the QAOA circuit: cir.state

    """

    cir = UAnsatz(N, input_state=input_state)
    # The first loop defines the QAOA circuit with P layers of two blocks
    for layer in range(P):
        # The second and third loops aim to construct the first block U_theta[layer][0] which involves
        # two-qubit operation e^{-i\beta Z_iZ_j} acting on a pair of qubits or nodes i and j in the circuit.
        for row in range(N):
            for col in range(N):
                if abs(adjacency_matrix[row, col]) and row < col:
                    cir.cnot([row + 1, col + 1])
                    cir.rz(
                        theta=theta[layer][0] * adjacency_matrix[row, col],
                        which_qubit=col + 1, )
                    cir.cnot([row + 1, col + 1])
        # This loops constructs the second block U_theta only involving the single-qubit operation e^{-i\beta X}.
        for i in range(1, N + 1):
            cir.rx(theta=theta[layer][1], which_qubit=i)

    return cir.state


def circuit_extend_QAOA(theta, input_state, adjacency_matrix, N, P):
    """
    This is an extended version of the QAOA circuit, and the main difference is U_theta[layer]([1]-[3]) constructed
    from the driving Hamiltonian describing the rotation around an arbitrary direction on each qubit.

    Args:
        theta: parameters to be optimized in the QAOA circuit
        input_state: input state of the QAOA circuit which usually is the uniform superposition of 2^N bit-strings
                     in the computational basis
        adjacency_matrix:  the adjacency matrix of the problem graph encoding the original problem
        N: number of qubits, or equivalently, the number of parameters in the original classical problem
        P: number of layers of two blocks in the QAOA circuit
    Returns:
        final state of the QAOA circuit: cir.state

    Note: If this U_extend_theta function is used to construct QAOA circuit, then we need to change the parameter layer
           in the Net function defined below from the Net(shape=[D, 2]) for U_theta function to Net(shape=[D, 4])
           because the number of parameters doubles in each layer in this QAOA circuit.
    """

    cir = UAnsatz(N, input_state=input_state)

    # The first loop defines the QAOA circuit with P layers of two blocks
    for layer in range(P):
        # The second and third loops aim to construct the first block U_theta[layer][0] which involves
        # two-qubit operation e^{-i\beta Z_iZ_j} acting on a pair of qubits or nodes i and j in the circuit.
        for row in range(N):
            for col in range(N):
                if abs(adjacency_matrix[row, col]) and row < col:
                    cir.cnot([row + 1, col + 1])
                    cir.rz(
                        theta=theta[layer][0] * adjacency_matrix[row, col],
                        which_qubit=col + 1, )
                    cir.cnot([row + 1, col + 1])
        # This loops constructs the second block U_theta[layer][1]-[3] composed of three single-qubit operation
        #  e^{-i\beta[1] Z}e^{-i\beta[2] X}e^{-i\beta[3] X} sequentially acting on single qubits.
        for i in range(1, N + 1):
            cir.rz(theta=theta[layer][1], which_qubit=i)
            cir.rx(theta=theta[layer][2], which_qubit=i)
            cir.rz(theta=theta[layer][3], which_qubit=i)

    return cir.state


class Net(fluid.dygraph.Layer):
    """
    It constructs the net for QAOA which combines the  QAOA circuit with the classical optimizer which sets rules
    to update parameters described by theta introduced in the QAOA circuit.

    """

    def __init__(
            self,
            shape,
            param_attr=fluid.initializer.Uniform(
                low=0.0, high=PI, seed=SEED),
            dtype="float32", ):
        super(Net, self).__init__()

        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    def forward(self, input_state, adjacency_matrix, out_state_store, N, P,
                METHOD):
        """
        This function constructs the loss function for the QAOA circuit.

        Args:
            self: the free parameters to be optimized in the QAOA circuit and defined in the above function
            input_state: initial state of the QAOA circuit which usually is the uniform superposition of 2^N bit-strings
                         in the computational basis $|0\rangle, |1\rangle$
            adjacency_matrix: the adjacency matrix generated from the graph encoding the classical problem
            out_state_store: the output state of the QAOA circuit
            N: number of qubits
            P: number of layers
            METHOD: which version of QAOA is chosen to solve the problem, i.e., standard version labeled by 1 or
            extended version by 2.
        Returns:
            The loss function for the parameterized QAOA circuit.
        """

        # Generate the problem_based quantum Hamiltonian H_problem based on the classical problem in paddle
        H, _ = H_generator(N, adjacency_matrix)
        H_problem = fluid.dygraph.to_variable(H)

        # The standard QAOA circuit: the function circuit_QAOA is used to construct the circuit, indexed by METHOD 1.
        if METHOD == 1:
            out_state = circuit_QAOA(self.theta, input_state, adjacency_matrix,
                                     N, P)
        # The extended QAOA circuit: the function circuit_extend_QAOA is used to construct the net, indexed by METHOD 2.
        elif METHOD == 2:
            out_state = circuit_extend_QAOA(self.theta, input_state,
                                            adjacency_matrix, N, P)
        else:
            raise ValueError("Wrong method called!")

        out_state_store.append(out_state.numpy())
        loss = pp_matmul(
            pp_matmul(out_state, H_problem),
            transpose(
                fluid.framework.ComplexVariable(out_state.real,
                                                -out_state.imag),
                perm=[1, 0], ), )

        return loss.real


def main(N=4):
    """
    This is the main function which maps the classical problem to the quantum version solved by QAOA and outputs
    the quantum solution and its corresponding classical ones. Here, N=4 is a 4-qubit example to show how QAOA works.

    """
    # Generate the adjacency matrix from the description of the problem-based graph
    _, classical_graph_adjacency = generate_graph(N, 1)
    Paddle_QAOA(classical_graph_adjacency)


def Paddle_QAOA(classical_graph_adjacency, N=4, P=4, METHOD=1, ITR=120,
                LR=0.1):
    """
    This is the core function to run QAOA.

     Args:
         classical_graph_adjacency: adjacency matrix to describe the graph which encodes the classical problem
         N: number of qubits (default value N=4)
         P: number of layers of blocks in the QAOA circuit (default value P=4)
         METHOD: which version of the QAOA circuit is used: 1, standard circuit (default); 2, extended circuit
         ITR: number of iteration steps for QAOA (default value ITR=120)
         LR: learning rate for the gradient-based optimization method (default value LR=0.1)
     Returns:
         optimized parameters theta and the bitstrings sampled from the output state with maximal probability
    """

    out_state_store = []
    with fluid.dygraph.guard():
        # Preparing the initial state
        _initial_state = ones([1, 2**N]).astype("complex64") / sqrt(2**N)
        initial_state = fluid.dygraph.to_variable(_initial_state)

        # Construct the net or QAOA circuits based on the standard modules
        if METHOD == 1:
            net = Net(shape=[P, 2])
        # Construct the net or QAOA circuits based on the extended modules
        elif METHOD == 2:
            net = Net(shape=[P, 4])
        else:
            raise ValueError("Wrong method called!")

        # Classical optimizer
        opt = fluid.optimizer.AdamOptimizer(
            learning_rate=LR, parameter_list=net.parameters())

        # Gradient descent loop
        summary_iter, summary_loss = [], []
        for itr in range(1, ITR + 1):
            loss = net(initial_state, classical_graph_adjacency,
                       out_state_store, N, P, METHOD)
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            print("iter:", itr, "  loss:", "%.4f" % loss.numpy())
            summary_loss.append(loss[0][0].numpy())
            summary_iter.append(itr)

        theta_opt = net.parameters()[0].numpy()
        print(theta_opt)

        os.makedirs("output", exist_ok=True)
        savez("./output/summary_data", iter=summary_iter, energy=summary_loss)

    # Output the measurement probability distribution sampled from the output state of optimized QAOA circuit.
    prob_measure = zeros([1, 2**N]).astype("complex")

    rho_out = out_state_store[-1]
    rho_out = np_matmul(conjugate(rho_out).T, rho_out).astype("complex")

    for index in range(0, 2**N):
        comput_basis = zeros([1, 2**N])
        comput_basis[0][index] = 1
        prob_measure[0][index] = real(
            np_matmul(np_matmul(comput_basis, rho_out), comput_basis.T))

    return prob_measure


if __name__ == "__main__":
    main()
