# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

r"""
The solver in the qchem.
"""

from typing import Optional, Tuple
import logging
import numpy as np
import paddle
from paddle.optimizer import Optimizer
from paddle_quantum.loss import ExpecVal
from paddle_quantum import Hamiltonian
from ..ansatz import Circuit
from ..state import State
from .molecule import Molecule

__all__ = ["VQESolver", "GroundStateSolver"]


class VQESolver(object):
    r"""
    VQE solver class.

    Args:
        optimizer: paddle optimizer.
        num_iterations: number of iterations during the optimization.
        tol: convergence criteria, if :math`|L1-L0|<tol` , optimization will stop.
        save_every: save loss value after ``save_every`` iterations.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_iterations: int,
        tol: float = 1e-8,
        save_every: int = 1,
    ) -> None:
        self.optimizer = optimizer
        self.num_iters = num_iterations
        self.tol = tol
        self.save_every = save_every

    def solve(self):
        raise NotImplementedError("Specific VQE solver should implement their own solve method.")


class GroundStateSolver(VQESolver):
    r"""
    The ground state solver class.

    Args:
        optimizer: paddle optimizer.
        num_iterations: number of iterations during the optimization.
        tol: convergence criteria, if :math:`|L1-L0|<tol` , optimization will stop.
        save_every: save loss value after ``save_every`` iterations.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_iterations: int,
        tol: float = 1e-8,
        save_every: int = 1,
    ) -> None:
        super().__init__(optimizer, num_iterations, tol, save_every)

    def solve(
        self,
        ansatz: Circuit,
        mol: Optional[Molecule] = None,
        ham: Optional[Hamiltonian] = None,
        init_state: Optional[State] = None,
        **optimizer_kwargs
    ) -> Tuple[float, paddle.Tensor]:
        r"""Run VQE to calculate the ground state energy for a given molecule.

        Args:
            ansatz (Circuit): the quantum circuit represents the wfn transformation.
            mol (Molecule): default is None, the molecule object.
            ham (Hamiltonian): default is None, the Hamiltonian on which to calculate the ground state energy.
            init_state (State): default is None, the initial state passed to the ansatz.
            optimizer_kwargs.

        Returns:
            The estimated ground state energy and the ground state wave function.
        """
        if mol is None:
            if ham is None:
                raise ValueError("`mol` and `ham` can't be None simultaneously.")
            elif isinstance(ham, Hamiltonian):
                num_qubits = ham.n_qubits
                h = ham
            else:
                raise ValueError("`ham` must be an instance of `paddle_quantum.Hamiltonian`.")
        elif isinstance(mol, Molecule):
            num_qubits = mol.num_qubits
            h = mol.get_molecular_hamiltonian()
        else:
            raise ValueError("`mol` must be an instance of `paddle_quantum.qchem.Molecule`")

        logging.info("\n#######################################\nVQE (Ground State)\n#######################################")
        logging.info(f"Number of qubits: {num_qubits:d}")
        logging.info(f"Ansatz: {ansatz.__class__.__name__:s}")
        logging.info(f"Optimizer: {self.optimizer.__name__:s}")

        optimizer = self.optimizer(parameters=ansatz.parameters(), **optimizer_kwargs)
        logging.info(f"\tlearning_rate: {optimizer.get_lr()}")

        energy_fn = ExpecVal(h)

        logging.info("\nOptimization:")
        loss0 = paddle.to_tensor(np.inf)
        for n in range(self.num_iters):
            intm_state: State = ansatz(init_state)
            loss = energy_fn(intm_state)

            with paddle.no_grad():
                if n % self.save_every == 0:
                    loss_v = loss.detach().item()
                    logging.info(f"\tItr {n:d}, loss={loss_v:.5f}.")
                # pay attention to the order of x and y !!!
                # see https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/isclose_en.html for details.
                if paddle.isclose(loss0, loss, atol=self.tol).item():
                    logging.info(f"Optimization converged after {n:d} iterations.")
                    loss_v = loss.detach().item()
                    logging.info(f"The final loss = {loss_v:.5f}.")
                    break

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss0 = loss

        with paddle.no_grad():
            final_state: State = ansatz(init_state)
            final_loss: float = final_state.expec_val(h)
        return final_loss, final_state
