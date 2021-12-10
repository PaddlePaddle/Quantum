# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
"""
利用不同的 ansatz 运行量子化学计算。
"""

from typing import List, Tuple, Callable
import numpy as np
import paddle
from paddle.optimizer import Optimizer, Adam
from . import qchem as pq_chem
import paddle_quantum.state as pq_state
from paddle_quantum.intrinsic import vec_expecval

from .ansatz.rhf import RestrictHartreeFockModel
from .qmodel import QModel
from .ansatz import HardwareEfficientModel


def _minimize(
        model: QModel,
        loss_fn: Callable[[QModel], Tuple[float, QModel]],
        optimizer: Optimizer,
        max_iters: int,
        a_tol: float
) -> Tuple[float, QModel]:
    loss_prev = -np.inf
    for i in range(max_iters):
        with paddle.no_grad():
            loss = loss_fn(model)
            print(f"Iteration {i+1:>d}, {model.__class__.__name__} energy {loss.item():>.5f}.")

            if np.abs(loss.item() - loss_prev) < a_tol:
                print(f"Converge after {(i+1):>d} number of iterations.")
                break
            else:
                loss_prev = loss.item()

        optimizer.clear_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()

    with paddle.no_grad():
        loss = loss_fn(model)

    return loss.item(), model


def run_chem(
        geometry: List[Tuple[str, List[float]]],
        ansatz: str,
        basis_set: str = "sto-3g",
        charge: int = 0,
        max_iters: int = 100,
        a_tol: float = 1e-6,
        optimizer_option: dict = {},
        ansatz_option: dict = {}
) -> Tuple[float, QModel]:
    r"""根据输入的分子信息进行量子化学计算。

    Args:
        geometry (List[Tuple[str, List[float]]]): 分子的几何结构，例如：[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.74])]。定义结构时使用的长度单位是 Angstrom。
        ansatz (str): 表示多体波函数的量子线路，目前我们支持 "hardware efficient" 和 "hartree fock"。
        basis_set (str): 用来展开原子轨道的量子化学的基函数，例如："sto-3g"。注意，复杂的基函数会极大占用计算资源。
        charge (int): 分子的电荷量。
        max_iters (int): 优化过程的最大迭代次数。
        a_tol (float): 优化收敛的判断依据，当 :math:`|e_{i+1} - e_i| < a_{tol}` 时，优化结束。
        optimizer_options (dict): 优化器的可选参数，例如：学习率 (learning_rate)、权重递减 (weight_decay)。
        ansatz_option (dict): 定义 ansatz 量子线路的可选参数。目前，对于 "hardware efficient" 方法，可选参数为 "cir_depth"（线路深度）。"hartree fock" 方法没有可选参数。

    Returns:
        tuple:
            - 基态能量
            - 优化后的 ansatz 量子线路
    """

    if ansatz == "hardware efficient":
        return run_hardware(
            geometry, basis_set, charge,
            max_iters, a_tol, optimizer_option, **ansatz_option
        )

    if ansatz == "hartree fock":
        return run_rhf(
            geometry, basis_set, charge,
            max_iters, a_tol, optimizer_option, **ansatz_option
        )

    else:
        raise NotImplementedError(
            """
            Currently, we only support "hardware efficient" or "hartree fock" for `ansatz` parameter, we will add more in the future. You can open an issue here 
            https://github.com/PaddlePaddle/Quantum/issues to report the ansatz you're interested in.
            """)


def run_hardware(
        geometry: List[Tuple[str, List[float]]],
        basis_set: str,
        charge: int,
        max_iters: int,
        a_tol: float,
        optimizer_option: dict = {},
        cir_depth: int = 3
) -> Tuple[float, QModel]:
    r"""hardware efficient 方法的 run 函数。

    """
    mol_data = pq_chem.get_molecular_data(geometry, basis=basis_set, charge=charge, if_print=False)
    mol_qubitH = pq_chem.spin_hamiltonian(mol_data)

    n_qubits = mol_qubitH.n_qubits
    ansatz = HardwareEfficientModel(n_qubits, cir_depth)
    optimizer = Adam(parameters=ansatz.parameters(), **optimizer_option)

    s0 = paddle.to_tensor(pq_state.vec(0, n_qubits))
    s0 = paddle.reshape(s0, [2**n_qubits])

    def loss_fn(model: QModel) -> paddle.Tensor:
        s = model(s0)
        return paddle.real(vec_expecval(mol_qubitH.pauli_str, s))

    mol_gs_en, updated_ansatz = _minimize(ansatz, loss_fn, optimizer, max_iters, a_tol)
    return mol_gs_en, updated_ansatz


def run_rhf(
        geometry: List[Tuple[str, List[float]]],
        basis_set: str,
        charge: int,
        max_iters: int,
        a_tol: float,
        optimizer_option: dict = {}
) -> Tuple[float, QModel]:
    r"""hartree fock 方法的 run 函数。
    """
    try:
        import pyscf
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            """
            Hartree Fock method needs `pyscf`,
            please run `pip install -U pyscf` to first install pyscf.
            """)

    from .molecule import get_molecular_information

    n_qubits, n_electrons, integrals = get_molecular_information(geometry, basis_set, charge)
    nuc_energy, onebody, twobody = [paddle.to_tensor(t) for t in integrals]
    ansatz = RestrictHartreeFockModel(n_qubits, n_electrons, onebody)
    optimizer = Adam(parameters=ansatz.parameters(), **optimizer_option)

    # run model to build the circuit
    bstr = "1"*n_electrons+"0"*(n_qubits-n_electrons)
    _s0 = paddle.to_tensor(pq_state.vec(int(bstr, 2), n_qubits))
    _s0 = paddle.reshape(_s0, [2**n_qubits])
    ansatz(_s0)

    def loss_fn(model: QModel) -> paddle.Tensor:
        U_hf = model.single_particle_U()
        rdm1 = U_hf @ U_hf.conj().t()
        return nuc_energy + 2*paddle.einsum("pq,qp->", onebody, rdm1)\
            + 4*paddle.einsum("pqrs,qp,sr->", twobody, rdm1, rdm1)\
            - 2*paddle.einsum("pqrs,sp,qr->", twobody, rdm1, rdm1)

    mol_gs_en, updated_ansatz = _minimize(ansatz, loss_fn, optimizer, max_iters, a_tol)
    return mol_gs_en, updated_ansatz
