# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
Paddle Quantum
==============

[Paddle Quantum](https://qml.baidu.com) is a quantum machine learning (QML) toolkit 
developed based on Baidu PaddlePaddle. It supports easy-to-use quantum neural 
networks (QNN) construction and training, provides combinatorial optimization, 
quantum chemistry and other cutting-edge quantum applications. 

See [online API](https://qml.baidu.com/api) for complete documentation.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from .backend import Backend
from .state import State
from .base import Operator
from .base import set_device, get_device
from .base import set_dtype, get_dtype
from .base import set_backend, get_backend
from .hamiltonian import Hamiltonian
from .ansatz import Circuit
from . import ansatz
from . import channel
from . import gate
from . import locc
from . import loss
from . import mbqc
from . import operator
from . import base
from . import dataset
from . import finance
from . import fisher
from . import gradtool
from . import hamiltonian
from . import linalg
from . import qinfo
from . import qml
from . import shadow
from . import trotter
from . import visual
from . import qchem

name = 'paddle_quantum'
__version__ = '2.4.0'
