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
Calculate the properties of the molecule.
"""

import warnings
import numpy as np
from ..state import State
from ..hamiltonian import Hamiltonian
from ..shadow import shadow_sample
from ..qinfo import shadow_trace
from .molecule import Molecule

__all__ = ["energy", "dipole_moment"]


def energy(
    psi: State,
    mol: Molecule,
    shots: int = 0,
    use_shadow: bool = False,
    **shadow_kwargs
) -> float:
    r"""Calculate the energy of a molecule w.r.t a quantum state :math:`\psi` .

    Args:
        psi: a quantum state.
        mol: the molecule instance.
        shots: number of shots used to estimate the expectation value, default is 0 and will calculate
            the ideal expectation value.
        use_shadow: whether use classical shadow to estimate the energy, default is False and will
            evalute the energy by matrix multiplication.
        shadow_kwargs

    Returns:
        The energy of the molecule.
    """
    warnings.warn("This method shouldn't be used as a loss function since it won't return tracked Tensor.")

    h = mol.get_molecular_hamiltonian()
    v = psi.expec_val(h, shots)
    return v if isinstance(v, float) else v.item()


def symmetric_rdm1e(psi: State, shots: int = 0, use_shadow: bool = False, **shadow_kwargs) -> np.ndarray:
    r"""Calculate the symmetric 1-RDM from a given quantum state :math:`\psi` .

    .. math::

        D_{pq}=<\hat{c}_p^{\dagger}\hat{c}_q+\hat{c}_{q}^{\dagger}\hat{c}_p>, p>q \\
        D_{pq}=0, p>q \\
        D_{pq}=<\hat{c}_p^{\dagger}\hat{c}_p>, p=q

    Args:
        psi: quantum state.
        shots:  number of shots used to estimate the expectation value. default is 0, and will calculate
            the ideal expectation value.
        use_shadow: whether use classical shadow to estimate the energy, default is False and will
            evalute the energy by matrix multiplication.
        **shadow_kwargs: The other args.

    Returns:
        The symmetric 1-RDM.
    """
    num_qubits = psi.num_qubits
    symm_rdm1 = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        diag_el = Hamiltonian([(0.5, "I"), (-0.5, f"Z{i:d}")])
        v = psi.expec_val(diag_el, shots)
        symm_rdm1[i, i] = v if isinstance(v, float) else v.item()
        for j in range(i+2, num_qubits, 2):
            qubit_op_str1 = f"X{i:d}, " + ", ".join(f"Z{k:d}" for k in range(i+2, j, 2)) + f"X{j:d}"
            qubit_op_str2 = f"Y{i:d}, " + ", ".join(f"Z{k:d}" for k in range(i+2, j, 2)) + f"Y{j:d}"
            offdiag_el = Hamiltonian([(0.5, qubit_op_str1), (0.5, qubit_op_str2)])
            v = psi.expec_val(offdiag_el, shots)
            symm_rdm1[j, i] = v if isinstance(v, float) else v.item()
    return symm_rdm1


def dipole_moment(psi: State, mol: Molecule, shots: int = 0, use_shadow: bool = False, **shadow_kwargs) -> np.ndarray:
    r"""Calculate the dipole moment of a molecule w.r.t a given quantum state.

    Args:
        psi: a quantum state.
        mol: the molecule instance.
        shots: number of shots used to estimate the expectation value. default is 0, and will calculate
            the ideal expectation value.
        use_shadow: whether use classical shadow to estimate the energy, default is False and will
            evalute the energy by matrix multiplication.
        **shadow_kwargs: The other args.

    Returns:
        The dipole moment of the input molecule.
    """
    warnings.warn("This method shouldn't be used as a loss function since it won't return tracked Tensor.")

    # get (p|x-R_c|q)
    int1e_r = mol.get_mo_integral("int1e_r")
    np.testing.assert_array_almost_equal(int1e_r, np.transpose(int1e_r, (0, 2, 1)))

    # nuclei dipole moment.
    charges = mol.atom_charges
    coords = mol.atom_coords
    nucl_dip = np.einsum('i,ix->x', charges, coords)

    # electron dipole moment.
    num_qubits = psi.num_qubits
    symm_rdm1 = symmetric_rdm1e(psi, shots, use_shadow, **shadow_kwargs)
    symm_rdm1_a = symm_rdm1[::2, ::2]
    symm_rdm1_b = symm_rdm1[1:num_qubits:2, 1:num_qubits:2]
    el_dip = -np.einsum("ijk,kj->i", int1e_r, (symm_rdm1_a+symm_rdm1_b))

    return nucl_dip + el_dip