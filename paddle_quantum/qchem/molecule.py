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
从分子结构等信息中提取量子化学需要的输入。
"""

from typing import Tuple, List
import numpy as np
from pyscf import gto, scf
from pyscf.lo import lowdin


# Transform one and two body integral into L\"owdin basis
def lowdin_transform(
        ovlp: np.array,
        onebody: np.array,
        twobody: np.array
) -> Tuple[np.array]:
    r"""该函数会将 pyscf 中得到的高斯积分利用 L\"owdin 正交化方法进行变换。

    Note:
        L\"owdin 正交化:
        :math:`S_{ij}=\langle\phi_i^{\text{G}}|\phi_j^{\text{G}}\rangle`,
        :math:`X = S^{-1/2}`

        Operators 的变换方式:
        :math:`A^{\prime}_{ij}=\sum_{pq}A_{pq}X_{ip}X_{qj}`
        :math:`B^{\prime}_{iklj}=\sum_{pqrs}B_{prsq}X_{ip}X_{kr}X_{sl}X_{qj}`

        分子轨道的变换方式:
        :math:`C^{\prime}_{ij}=\sum_{p}C_{pj}X^{-1}_{ip}`

    Args:
        ovlp (np.array): 交叠积分，`mol.intor("int1e_ovlp")`。
        onebody (np.array): 单体积分，`mol.intor("int1e_kin")+mol.intor("int1e_nuc")`。
        twobody (np.array): 两体积分，`mol.intor("int2e")`。

    Returns:
        Tuple[np.array]:
            - 变换之后的单体积分。
            - 变换之后的两体积分。
    """

    inv_half_ovlp = lowdin(ovlp)
    t_onebody = inv_half_ovlp @ onebody @ inv_half_ovlp
    t_twobody = np.einsum(
        "pqrs,ip,qj,kr,sl->ijkl", 
        twobody, inv_half_ovlp, inv_half_ovlp, inv_half_ovlp, inv_half_ovlp
        )
    return t_onebody, t_twobody


# Molecular information
def get_molecular_information(
        geometry: List[Tuple[str, List]],
        basis: str = "sto-3g",
        charge: int = 0,
        debug: bool = False
) -> Tuple:
    r"""该函数会返回量子化学（目前是 "hartree fock" 方法）计算所需要的分子信息，包括有计算需要的量子比特的数量，分子中的电子数，分子积分和 pyscf 平均场结果 (optional)。

    Args:
        geometry (List[Tuple[str,List]]): 分子中各原子笛卡尔坐标；
        basis (str): 基函数名称，例如："sto-3g"；
        charge (int): 分子的电荷；
        debug (bool): 是否使用 debug 模式。debug 模式会返回 pyscf 的平均场计算结果。

    Returns:
        Tuple:
            - 量子比特数目；
            - 分子中的电子数；
            - 原子核之间的排斥能、单体积分、双体积分；
            - pyscf 的平均场计算结果 (可选，只有 debug=True 才会返回)。
    """
    mol = gto.M(atom=geometry, basis=basis, charge=charge, unit="angstrom")
    mol.build()
    if debug:
        mf_mol = scf.RHF(mol).run()

    int_ovlp = mol.intor("int1e_ovlp")
    nuc_energy = mol.energy_nuc()
    onebody = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    twobody = mol.intor("int2e")

    orth_onebody, orth_twobody = lowdin_transform(int_ovlp, onebody, twobody)

    if debug:
        return 2*mol.nao, mol.nelectron, (nuc_energy, orth_onebody, 0.5*orth_twobody), mf_mol
    else:
        return 2*mol.nao, mol.nelectron, (nuc_energy, orth_onebody, 0.5*orth_twobody)
