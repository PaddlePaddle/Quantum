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
The source file of the class for quantum circuit templates.
"""

import matplotlib
import numpy as np
from typing import Iterable, List, Union, Tuple, Dict

from .container import Sequential
from ..gate import H, RX, RY, RZ, U3, CNOT
from ..intrinsic import _cnot_idx_fetch, _inverse_gather_for_dm


__all__ = ['Layer', 'SuperpositionLayer', 'LinearEntangledLayer', 'RealEntangledLayer', 'ComplexEntangledLayer',
           'RealBlockLayer', 'QAOALayer', 'QAOALayerWeighted']


def _qubits_idx_filter(qubits_idx: Union[List[int], str], num_qubits: int) -> List[int]:
    r"""Check the validity of ``qubits_idx`` and ``num_qubits``.

    Args:
        qubits_idx: Indices of qubits.
        num_qubits: Total number of qubits.

    Raises:
        RuntimeError: You must specify ``qubits_idx`` or ``num_qubits`` to instantiate the class.
        ValueError: The ``qubits_idx`` must be ``Iterable`` or ``None``.

    Returns:
        Checked indices of qubits.
    """
    if qubits_idx is None or qubits_idx == 'full':
        if num_qubits is None:
            raise RuntimeError(
                "You must specify qubits_idx or num_qubits to instantiate the class.")
        return list(range(num_qubits))
    elif isinstance(qubits_idx, Iterable):
        assert len(np.array(qubits_idx).shape) == 1, \
            "The input qubit index must be a list of int for layers."
        qubits_idx = list(qubits_idx)
        assert len(qubits_idx) > 1, \
            f"Requires more than 1 qubit for a layer to act on: received length {len(qubits_idx)}"
        assert len(qubits_idx) == len(set(qubits_idx)), \
            f"Layers do not allow repeated indices: received {qubits_idx}"
        return qubits_idx
    
    raise ValueError(f"The qubits_idx must be a list of int or None: received {type(qubits_idx)}")


class Layer(Sequential):
    r"""Base class for Layers.
    
    Args:
        qubits_idx: Indices of the qubits on which this layer is applied.
        num_qubits: Total number of qubits.
        depth: Number of layers.
        
    Note:
        A Circuit instance needs to extend this Layer instance to be used in a circuit. 
    
    """
    def __init__(self, qubits_idx: Union[Iterable[int], str], num_qubits: int, depth: int = 1):
        self.num_qubits = num_qubits
        self.qubits_idx = _qubits_idx_filter(qubits_idx, num_qubits)
        self.depth = depth
        super().__init__()
        
    @property
    def gate_history(self):
        r""" list of gates information of this layer

        Returns:
            history of quantum gates
        """
        gate_history = []
        for op in self.sublayers():
            if op.gate_info['gatename'] is None:
                raise NotImplementedError(
                    f"{type(op)} has no gate name and hence cannot be recorded into history.")
            op.gate_history_generation()
            gate_history.extend(op.gate_history)
        return gate_history


class SuperpositionLayer(Layer):
    r"""Layers of Hadamard gates.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1
    ):
        super().__init__(qubits_idx, num_qubits, depth)
        self.append(H(self.qubits_idx, num_qubits, depth))


class WeakSuperpositionLayer(Layer):
    r"""Layers of Ry gates with a rotation angle :math:`\pi/4`.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1
    ):
        super().__init__(qubits_idx, num_qubits, depth)
        self.append(RY(self.qubits_idx, num_qubits, depth, np.pi / 4, param_sharing=True))


class LinearEntangledLayer(Layer):
    r"""Linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1
    ):
        super().__init__(qubits_idx, num_qubits, depth)
        assert len(self.qubits_idx)>1, 'acted qubits need more than 1'
        num_acted_qubits = len(self.qubits_idx)
        acted_list = [(self.qubits_idx[idx], self.qubits_idx[idx+1]) for idx in range(num_acted_qubits-1)]
        cnot_idx = _cnot_idx_fetch(num_qubits=self.num_qubits, qubits_idx=acted_list)

        for _ in range(self.depth):
            self.append(RY(self.qubits_idx))
            self.append(CNOT(qubits_idx=acted_list, cnot_idx=cnot_idx))
            self.append(RZ(self.qubits_idx))
            self.append(CNOT(qubits_idx=acted_list, cnot_idx=cnot_idx))


class RealEntangledLayer(Layer):
    r"""Strongly entangled layers consisting of Ry gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a real unitary matrix.
        This ansatz is from the following paper: https://arxiv.org/pdf/1905.10876.pdf.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1
    ):
        super().__init__(qubits_idx, num_qubits, depth)
        assert len(self.qubits_idx)>1, 'acted qubits need more than 1'
        acted_list = [(self.qubits_idx[idx], self.qubits_idx[(idx + 1) % len(self.qubits_idx)])
                      for idx in range(len(self.qubits_idx))]
        cnot_idx = _cnot_idx_fetch(num_qubits=self.num_qubits, qubits_idx=acted_list)

        for _ in range(self.depth):
            self.append(RY(self.qubits_idx))
            self.append(CNOT(qubits_idx=acted_list, cnot_idx=cnot_idx))
            

class ComplexEntangledLayer(Layer):
    r"""Strongly entangled layers consisting of single-qubit rotation gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a complex unitary matrix.
        This ansatz is from the following paper: https://arxiv.org/abs/1804.00633.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1
    ):
        super().__init__(qubits_idx, num_qubits, depth)
        assert len(self.qubits_idx)>1, 'acted qubits need more than 1'
        acted_list = [(self.qubits_idx[idx], self.qubits_idx[(idx + 1) % len(self.qubits_idx)])
                      for idx in range(len(self.qubits_idx))]
        cnot_idx = _cnot_idx_fetch(num_qubits=self.num_qubits, qubits_idx=acted_list)

        for _ in range(self.depth):
            self.append(U3(self.qubits_idx))
            self.append(CNOT(qubits_idx=acted_list, cnot_idx=cnot_idx))


class RealBlockLayer(Layer):
    r"""Weakly entangled layers consisting of Ry gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a real unitary matrix.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1
    ):
        super().__init__(qubits_idx, num_qubits, depth)
        assert len(self.qubits_idx)>1, 'acted qubits need more than 1'
        num_acted_qubits = len(self.qubits_idx)
        if num_acted_qubits % 2 == 0:
            for _ in range(self.depth):
                self.__add_real_layer([0, num_acted_qubits - 1])
                self.__add_real_layer([1, num_acted_qubits - 2]) if num_acted_qubits > 2 else None
        else:
            for _ in range(self.depth):
                self.__add_real_layer([0, num_acted_qubits - 2])
                self.__add_real_layer([1, num_acted_qubits - 1])

    def __add_real_layer(self, position: List) -> None:
        cnot_acted_list = [[self.qubits_idx[i], self.qubits_idx[i+1]] for i in range(position[0], position[1], 2)]
        ry_acted_list = []
        for i in range(position[0], position[1], 2):
            ry_acted_list.extend((self.qubits_idx[i], self.qubits_idx[i+1]))
        self.append(RY(ry_acted_list))
        self.append(CNOT(cnot_acted_list))
        self.append(RY(ry_acted_list))


class ComplexBlockLayer(Layer):
    r"""Weakly entangled layers consisting of single-qubit rotation gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a complex unitary matrix.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied. 
        Defaults to ``None`` i.e. applied on all qubits.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(self, qubits_idx: Union[Iterable[int], str] = None, num_qubits: int = None, depth: int = 1):
        super().__init__(qubits_idx, num_qubits, depth)
        assert len(self.qubits_idx)>1, 'acted qubits need more than 1'
        num_acted_qubits = len(self.qubits_idx)
        if num_acted_qubits % 2 == 0:
            for _ in range(self.depth):
                self.__add_complex_layer([0, num_acted_qubits - 1])
                if num_acted_qubits > 2:
                    self.__add_complex_layer([1, num_acted_qubits - 2])
        else:
            for _ in range(self.depth):
                self.__add_complex_layer([0, num_acted_qubits - 2])
                self.__add_complex_layer([1, num_acted_qubits - 1])
    
    def __add_complex_layer(self, position: List[int]) -> None:
        cnot_acted_list = [[self.qubits_idx[i], self.qubits_idx[i+1]] for i in range(position[0], position[1], 2)]
        u3_acted_list = []
        for i in range(position[0], position[1], 2):
            u3_acted_list.extend((self.qubits_idx[i], self.qubits_idx[i+1]))
        self.append(U3(u3_acted_list))
        self.append(CNOT(cnot_acted_list))
        self.append(U3(u3_acted_list))


class QAOALayer(Layer):
    r""" QAOA driving layers
    
    Note:
        this layer only works for MAXCUT problem
    
    Args:
        edges: edges of the graph
        nodes: nodes of the graph
        depth: depth of layer
    
    """
    # TODO: only for maxcut now
    def __init__(
            self, edges: Iterable, nodes: Iterable, depth: int = 1
    ):
        Sequential.__init__(self)
        self.edges, self.nodes, self.depth = edges, nodes, depth
        
        for _ in range(self.depth):
            for node0, node1 in self.edges:
                self.append(CNOT([node0, node1]))
                self.append(RZ(node1))
                self.append(CNOT([node0, node1]))
            self.append(RX(self.nodes))


class QAOALayerWeighted(Layer):
    r""" QAOA driving layers with weights
    
    Args:
        edges: edges of the graph with weights
        nodes: nodes of the graph with weights
        depth: depth of layer
    
    """
    def __init__(
        self, edges: Dict[Tuple[int, int], float], nodes: Dict[int, float], depth: int = 1
    ) -> None:
        Sequential.__init__(self)
        self.edges, self.nodes, self.depth = edges.keys(), nodes.keys(), depth
        
        for _ in range(self.depth):
            for node0, node1 in self.edges:
                self.append(CNOT([node0, node1]))
                self.append(RZ(node1))
                self.append(CNOT([node0, node1]))
            self.append(RZ(self.nodes))
            self.append(RX(self.nodes))
