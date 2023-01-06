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
various indicators used to build the protein Hamiltonian
"""


from typing import Dict, Tuple, Optional, List
from openfermion import QubitOperator

__all__ = ["edge_direction_indicator", "contact_indicator", "backwalk_indicator"]


def edge_direction_indicator(
        edge: Tuple[int],
        affected_qubits: Optional[List[int]] = None,
        direction: Optional[int] = None
) -> Tuple[float, Dict]:
    r"""Calculate the direction indicate operator for a given edge at given
    affected qubits.

    .. math::

        I_0(e)=(1-q_0)(1-q_1) \\
        I_1(e)=(1-q_0)q_1 \\
        I_2(e)=q_0(1-q_1) \\
        I_3(e)=q_0q_1 \\
        satisfies \sum_{k} I_k(e) == 1

    Args:
        edge: Edge index in protein's directed graph.
        affected_qubits: The indices of qubits used to encode the indicator.
        direction: Direction of edge in the diamond lattice, valid values 0, 1, 2, 3.
    
    Returns:
        A tuple contains the sign and indicator operator corresponds to that edge.
    """
    indicators = {}
    if isinstance(direction, int):
        for i in range(4):
            if i != direction:
                indicators[i] = QubitOperator("", 0.0)
            else:
                indicators[i] = QubitOperator("", 1.0)
    elif isinstance(affected_qubits, list):
        qa, qb = affected_qubits
        one_plus_za = 0.5*QubitOperator("") + 0.5*QubitOperator(f"Z{qa:d}")
        one_plus_zb = 0.5*QubitOperator("") + 0.5*QubitOperator(f"Z{qb:d}")
        one_minus_za = 0.5*QubitOperator("") - 0.5*QubitOperator(f"Z{qa:d}")
        one_minus_zb = 0.5*QubitOperator("") - 0.5*QubitOperator(f"Z{qb:d}")
        indicators[0] = (one_plus_za*one_plus_zb)
        indicators[1] = (one_plus_za*one_minus_zb)
        indicators[2] = (one_minus_za*one_plus_zb)
        indicators[3] = (one_minus_za*one_minus_zb)
    else:
        raise ValueError("One of the `affected_qubits` and `direction` kwargs must be specified.")
    return (-1)**edge[0], indicators


def contact_indicator(qindex: int) -> QubitOperator:
    r"""The indicator which indicates whether two nodes in the protein are contact.

    .. math::

        qindex = contactor_start + index_of_contact_pair

    Args:
        qindex : index of qubit used as indicator of whether two nodes in the protein chain contact.
    
    Returns:
        Contact indicator in QubitOperator form.
    """
    return QubitOperator("", 0.5) - QubitOperator(f"Z{qindex:d}", 0.5)


def backwalk_indicator(e0_attrs: Dict, e1_attrs: Dict) -> QubitOperator:
    r"""Indicator of whether two consecutive bonds in protein overlap.

    .. math::

        \sum_{a=0}^3 I_a(e_{i})I_a(e_{i+1})

    Args:
        e0_attrs: Attributes of e0 edge.
        e1_attrs : Attributes of e1 (edge next to e0) edge.
    
    Returns:
        Backwalk indicator in QubitOperator form.
    """
    h = 0.0
    for i in range(4):
        h += e0_attrs[i]*e1_attrs[i]
    return h
