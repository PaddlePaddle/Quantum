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
The source file of the Sequential class.
"""

import collections
from paddle_quantum import Operator
from typing import Optional, Union, Iterable, Any, List


class Sequential(Operator):
    r"""Sequential container.

    Args:
        *operators: initial operators ready to be a sequential
    
    Note:
        Sublayers will be added to this container in the order of argument in the constructor.
        The argument passed to the constructor can be iterable Layers or iterable name Layer pairs.
    
    """
    def __init__(self, *operators: Operator):
        super().__init__()
        self.index = 0
        if len(operators) > 0 and isinstance(operators[0], (list, tuple)):
            for name, oper in operators:
                self.add_sublayer(name, oper)
        else:
            for idx, oper in enumerate(operators):
                self.add_sublayer(str(idx), oper)

    def __getitem__(self, name: Union[str, slice]) -> Operator:
        if isinstance(name, slice):
            return self.__class__(*(list(self._sub_layers.values())[name]))
        if isinstance(name, str):
            return self._sub_layers[name]
        if name >= len(self._sub_layers):
            raise IndexError(f'index {name:s} is out of range')
        if 0 > name >= -len(self._sub_layers):
            name += len(self._sub_layers)
        elif name < -len(self._sub_layers):
            raise IndexError(f'index {name:s} is out of range')
        return self._sub_layers[str(name)]

    def __setitem__(self, name: Any, layer: Operator) -> None:
        assert isinstance(layer, Operator)
        setattr(self, str(name), layer)

    def __delitem__(self, name: Any) -> None:
        name = str(name)
        assert name in self._sub_layers
        del self._sub_layers[name]

    def __iter__(self):
        return self

    def __next__(self) -> Union[Operator, StopIteration]:
        if self.index < len(self._sub_layers):
            oper = self._sub_layers[str(self.index)]
            self.index += 1
            return oper
        self.index = 0
        raise StopIteration

    def __len__(self):
        return len(self._sub_layers)

    def append(self, operator: Union[Iterable, Operator]) -> None:
        r""" append an operator
        
        Args:
            operator: operator with a name or just an operator
        
        """
        if isinstance(operator, Operator):
            idx = len(self._sub_layers)
            self.add_sublayer(str(idx), operator)
        elif isinstance(operator, Iterable):
            name, oper = operator
            self.add_sublayer(name, oper)


    def extend(self, operators: List[Operator]) -> None:
        r""" append a list of operators
        
        Args:
            operator: list of operators
        
        """
        if len(operators) > 0 and isinstance(operators[0], (list, tuple)):
            for name, oper in operators:
                self.add_sublayer(name, oper)
        else:
            origin_len = len(self._sub_layers)
            for idx, oper in enumerate(operators):
                self.add_sublayer(str(idx + origin_len), oper)

    def insert(self, index: int, operator: Operator) -> None:
        r""" insert an operator at ``index``
        
        Args:
            index: index to be inserted
            operator: an operator
        
        """
        new_operators = collections.OrderedDict()
        assert index <= len(self._sub_layers), 'the index ' + str(index) + ' should be no more than ' + str(len(self._sub_layers))
        if index == len(self._sub_layers):
            self.append(operator)
        for idx, name in enumerate(self._sub_layers):
            if idx < index:
                new_operators[name] = self._sub_layers[name]
            elif idx == index:
                if isinstance(operator, (list, tuple)):
                    name, oper = operator
                    new_operators[name] = oper
                elif isinstance(operator, Operator):
                    new_operators[str(index)] = operator
                if name.isdigit():
                    new_operators[str(int(name) + 1)] = self._sub_layers[name]
                else:
                    new_operators[name] = self._sub_layers[name]
            elif name.isdigit():
                new_operators[str(int(name) + 1)] = self._sub_layers[name]
            else:
                new_operators[name] = self._sub_layers[name]
        self._sub_layers = new_operators

    def pop(self, index: int = None, operator:  Optional[Operator] = None):
        r""" remove the operator at ``index`` or matched with ``operator``
        
        Args:
            index: at which the operator will be popped
            operator: matched with which the operator will be popped
        
        """
        if index is not None:
            assert index < len(self._sub_layers), 'the index ' + str(index) + ' should be less than ' + str(len(self._sub_layers))
            if isinstance(index, int):
                index = str(index)
            operator = self._sub_layers[index]
        if operator is None:
            raise ValueError("The index or operator must be input.")
        new_operators = collections.OrderedDict()
        behind_operator = False
        for name in self._sub_layers:
            if operator is self._sub_layers[name]:
                behind_operator = True
            elif not behind_operator:
                new_operators[name] = self._sub_layers[name]
            elif name.isdigit():
                new_operators[str(int(name) - 1)] = self._sub_layers[name]
            else:
                new_operators[name] = self._sub_layers[name]
        self._sub_layers = new_operators

    def forward(self, state: Any) -> Any:
        r""" forward the input
        
        Args:
            state: initial state
            
        Returns:
            output state
        
        """        
        for layer in self._sub_layers.values():
            state = layer(state)
        return state
