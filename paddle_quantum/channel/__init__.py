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
The module of the quantum channels.
"""

from .base import Channel
from .common import BitFlip
from .common import PhaseFlip
from .common import BitPhaseFlip
from .common import AmplitudeDamping
from .common import GeneralizedAmplitudeDamping
from .common import PhaseDamping
from .common import Depolarizing
from .common import PauliChannel
from .common import ResetChannel
from .common import ThermalRelaxation
from .custom import KrausRepr
from .custom import ChoiRepr
