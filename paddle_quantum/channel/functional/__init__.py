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
The module that contains the functions of various quantum channels.
"""

from .common import bit_flip
from .common import phase_flip
from .common import bit_phase_flip
from .common import amplitude_damping
from .common import generalized_amplitude_damping
from .common import phase_damping
from .common import depolarizing
from .common import pauli_channel
from .common import reset_channel
from .common import thermal_relaxation
from .common import kraus_repr
from .common import choi_repr
