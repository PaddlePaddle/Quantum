# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
VQE algorithm to solve protein folding problem
"""

from typing import Tuple, List
import logging
import math
import numpy as np
import paddle
from paddle.optimizer import Optimizer
from paddle_quantum.ansatz import Circuit
from paddle_quantum import Hamiltonian
from paddle_quantum.loss import ExpecVal
from paddle_quantum.state import State, computational_basis
from paddle_quantum.qchem.algorithm import VQESolver
from paddle_quantum.biocomputing import Protein

__all__ = ["ProteinFoldingSolver"]


# TODO: add this function to a method of State!
def cvar_expectation(psi: State, h: Hamiltonian, alpha: float) -> paddle.Tensor:
    r"""Calculate CVaR expectation value

    .. math::

        \sum_{i<=j} p_i \le alpha
        (1/\alpha) * (\sum_{i<j} p_i*H_i + H_j*(\alpha - \sum_{i<j} p_i))
    
    Args:
        psi: Quantum state.
        h: Hamiltonian with which to calculate expectation value.
        alpha: Parameter controls the number of basis states included in the evaluation of expectation value.
    
    Returns:
        The CVaR expectation value.
    """
    assert alpha > 0 and alpha <= 1, "alpha must in (0, 1]."

    if math.isclose(alpha, 1.0):
        return ExpecVal(h)(psi)

    probabilities = paddle.real(paddle.multiply(psi.data.conj(), psi.data))
    num_qubits = psi.num_qubits
    energies = []
    for i in range(2**num_qubits):
        phi = computational_basis(num_qubits, i)
        energies.append(phi.expec_val(h))
    results = sorted(zip(energies, probabilities))

    running_probability = 0
    cvar_val0 = 0
    cutoff_idx = 0
    for e, p in results:
        if running_probability >= alpha:
            break
        running_probability += p
        cvar_val0 += e*p
        cutoff_idx += 1
    Hj = results[cutoff_idx][0]
    return (1/alpha)*(cvar_val0 + Hj*(alpha - running_probability))


class ProteinFoldingSolver(VQESolver):
    r"""
    Args:
        penalty_factors: penalty factor ``[lambda0, lambda1]`` used in building the protein's Hamiltonian.
        alpha: the cutoff probability for CVaR expectation value calculation.
        optimizer: paddle's optimizer used to perform parameter update.
        num_iterations : number of VQE iterations.
        tol: convergence criteria.
        save_every : number of steps between two recorded VQE loss.
    """
    def __init__(
            self,
            penalty_factors: List[float],
            alpha: float,
            optimizer: Optimizer,
            num_iterations: int,
            tol: float = 1e-8,
            save_every: int = 1,
    ) -> None:
        super().__init__(optimizer, num_iterations, tol, save_every)
        self.lambda0 = penalty_factors[0]
        self.lambda1 = penalty_factors[1]
        self.alpha = alpha

    def solve(self, protein: Protein, ansatz: Circuit, **optimizer_kwargs) -> Tuple[float, str]:
        r"""Run VQE to calculate the structure of the protein that satisfies various constraints.

        Args:
            protein: the protein structure to be optimized.
            ansatz: the quantum circuit represents the unitary transformation.
            optimizer_kwargs: see PaddlePaddle's optimizer API for details https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html (Chinese) or https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/optimizer/Optimizer_en.html (English).

        Returns:
            A tuple contains the final loss value and optimal basis state.
        """
        h = protein.get_protein_hamiltonian(self.lambda0, self.lambda1, 1.0)

        logging.info("\n#######################################\nVQE (Protein Folding)\n#######################################")
        logging.info(f"Number of qubits: {h.n_qubits:d}")
        logging.info(f"Ansatz: {ansatz.__class__.__name__:s}")
        logging.info(f"Optimizer: {self.optimizer.__name__:s}")

        optimizer = self.optimizer(parameters=ansatz.parameters(), **optimizer_kwargs)
        logging.info(f"\tlearning_rate: {optimizer.get_lr()}")

        logging.info("\nOptimization:")
        loss0 = paddle.to_tensor(np.inf)
        for n in range(self.num_iters):
            intm_state: State = ansatz()
            loss = cvar_expectation(intm_state, h, self.alpha)

            with paddle.no_grad():
                if n % self.save_every == 0:
                    loss_v = loss.detach().item()
                    logging.info(f"\tItr {n:d}, loss={loss_v:.5f}.")
                # pay attention to the order of x and y !!!
                # see https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/isclose_en.html for details.
                if paddle.isclose(loss0, loss, atol=self.tol).item():
                    logging.info(f"Optimization converged after {n:d} iterations.")
                    break

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss0 = loss

        with paddle.no_grad():
            final_state: State = ansatz()
            final_loss = cvar_expectation(final_state, h, self.alpha)
            logging.info(f"The final loss = {final_loss.item():.5f}.")
            results = final_state.measure()
            sol_str = max(results.items(), key=lambda x: x[1])[0]
            logging.info(f"The solution is {sol_str}.")
        return final_loss.item(), sol_str
