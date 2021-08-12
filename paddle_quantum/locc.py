# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

import warnings
import numpy as np
import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import NKron, partial_trace, dagger
from paddle import matmul, trace, divide, kron, add, multiply
from paddle import sin, cos, real
from math import log2, sqrt


class LoccStatus(object):
    r"""用于表示 LOCCNet 中的一个 LOCC 态节点。
    
    由于我们在 LOCC 中不仅关心量子态的解析形式，同时还关心得到它的概率，以及是经过怎样的测量而得到。因此该类包含三个成员变量：量子态 ``state`` 、得到这个态的概率 ``prob`` 和得到这个态的测量的测量结果是什么，即 ``measured_result`` 。

    Attributes:
        state (paddle.Tensor): 表示量子态的矩阵形式
        prob (paddle.Tensor): 表示得到这个量子态的概率
        measured_result (str): 表示得到这个态的测量函数的测量结果
    """

    def __init__(self, state=None, prob=None, measured_result=None):
        r"""构造函数，用于实例化一个 ``LoccStatus`` 对象。

        Args:
            state (paddle.Tensor): 默认为 ``None`` ，该 ``LoccStatus`` 的量子态的矩阵形式
            prob (paddle.Tensor): 默认为 ``None`` ，得到该量子态的概率
            measured_result (str): 默认为 ``None`` ，表示得到这个态的测量函数的测量结果
        """
        super(LoccStatus, self).__init__()
        if state is None and prob is None and measured_result is None:
            self.state = paddle.to_tensor(np.array([1], dtype=np.complex128))
            self.prob = paddle.to_tensor(np.array([1], dtype=np.float64))
            self.measured_result = ""
        else:
            self.state = state
            self.prob = prob
            self.measured_result = measured_result

    def clone(self):
        r"""创建一个当前对象的副本。

        Returns:
            LoccStatus: 当前对象的副本
        """
        return LoccStatus(self.state, self.prob, self.measured_result)

    def __getitem__(self, item):
        if item == 0:
            return self.state
        elif item == 1:
            return self.prob
        elif item == 2:
            return self.measured_result
        else:
            raise ValueError("too many values to unpack (expected 3)")
    
    def __repr__(self):
        return f"state: {self.state.numpy()}\nprob: {self.prob.numpy()[0]}\nmeasured_result: {self.measured_result}"

    def __str__(self):
        return f"state: {self.state.numpy()}\nprob: {self.prob.numpy()[0]}\nmeasured_result: {self.measured_result}"


class LoccParty(object):
    r"""LOCC 的参与方。

    Attributes:
        qubit_number (int): 参与方的量子比特数量
    """

    def __init__(self, qubit_number):
        r"""
        构造函数，用于实例化一个 ``LoccParty`` 对象。

        Args:
            qubit_number (int): 参与方的量子比特数量
        """
        super(LoccParty, self).__init__()
        self.qubit_number = qubit_number
        self.qubits = [None] * qubit_number

    def __setitem__(self, key, value):
        self.qubits[key] = value

    def __getitem__(self, item):
        return self.qubits[item]

    def __len__(self):
        return self.qubit_number


class LoccAnsatz(UAnsatz):
    r"""继承 ``UAnsatz`` 类，目的是建立在 LOCC 任务上的电路模板。
    
    在 LOCC 任务中，每一方参与者只能在自己的量子比特上进行量子操作。因此我们只允许在每一方的量子比特上添加本地电路门。

    Attributes:
        party (LoccParty): 参与方
        m (int): 参与方的量子比特数量
    """

    def __init__(self, party, n):
        r"""构造函数，用于实例化一个 ``LoccAnsatz`` 的对象。

        Args:
            party (LoccParty): 参与方
            n (int): 全局的量子比特数量
        """
        super(LoccAnsatz, self).__init__(n)
        self.party = party
        self.m = len(self.party)
        self.measure = None
        self.expecval = None

    def run(self, status=None):
        r"""运行当前添加的电路门，并获得运行后的 LOCC 态节点。

        Args:
            status (LoccStatus or list): 作为LOCC下的量子电路的输入的 LOCC 态节点，其类型应该为 ``LoccStatus`` 或由其组成的 ``list``

        Returns:
            LoccStatus or list: 量子电路运行后得到的 LOCC 态节点,类型为 ``LoccStatus`` 或由其组成的 ``list``
        """
        if isinstance(status, LoccStatus):
            assert int(log2(sqrt(status.state.numpy().size))) == self.n, "the length of qubits should be same"
            state = super(LoccAnsatz, self).run_density_matrix(status.state)
            new_status = LoccStatus(state, status.prob, status.measured_result)
        elif isinstance(status, list):
            assert int(log2(sqrt(status[0].state.numpy().size))) == self.n, "the length of qubits should be same"
            new_status = list()
            for each_status in status:
                state = super(LoccAnsatz, self).run_density_matrix(each_status.state)
                new_status.append(LoccStatus(state, each_status.prob, each_status.measured_result))
        else:
            raise ValueError("can't recognize the input status")

        return new_status

    def rx(self, theta, which_qubit):
        r"""添加关于 x 轴的单量子比特旋转门。

        Args:
            theta (Tensor): 量子门的角度
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).rx(theta, which_qubit)

    def ry(self, theta, which_qubit):
        r"""添加关于 y 轴的单量子比特旋转门。

        Args:
            theta (Tensor): 量子门的角度
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).ry(theta, which_qubit)

    def rz(self, theta, which_qubit):
        r"""添加关于 z 轴的单量子比特旋转门。

        Args:
            theta (Tensor): 量子门的角度
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).rz(theta, which_qubit)

    def cnot(self, control):
        r"""添加一个 CNOT 门。

        Args:
            control (list): 作用在的 qubit 的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，其值都应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        control = [self.party[which_qubit] for which_qubit in control]
        super(LoccAnsatz, self).cnot(control)

    def swap(self, control):
        r"""添加一个 SWAP 门。

        Args:
            control (list): 作用在的 qubit 的编号，``control[0]`` 和 ``control[1]`` 是想要交换的位，其值都应该在 :math:`[0, m)`范围内， :math:`m` 为该参与方的量子比特数
        """
        control = [self.party[which_qubit] for which_qubit in control]
        super(LoccAnsatz, self).swap(control)

    def x(self, which_qubit):
        r"""添加单量子比特 X 门。

        Args:
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).x(which_qubit)

    def y(self, which_qubit):
        r"""添加单量子比特 Y 门。

        Args:
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).y(which_qubit)

    def z(self, which_qubit):
        r"""添加单量子比特 Z 门。

        Args:
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).z(which_qubit)

    def h(self, which_qubit):
        r"""添加单量子比特 Hadamard 门。

        Args:
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).h(which_qubit)

    def s(self, which_qubit):
        r"""添加单量子比特 S 门。

        Args:
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).s(which_qubit)

    def t(self, which_qubit):
        r"""添加单量子比特 T 门。

        Args:
            which_qubit (int): 添加该门量子比特编号
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).t(which_qubit)

    def u3(self, theta, phi, lam, which_qubit):
        r"""添加一个单量子比特的旋转门。

        Args:
            theta (Tensor): 旋转角度 :math:`\theta` 。
            phi (Tensor): 旋转角度 :math:`\phi` 。
            lam (Tensor): 旋转角度 :math:`\lambda` 。
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该量子电路的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).u3(theta, phi, lam, which_qubit)

    def universal_2_qubit_gate(self, theta, which_qubits):
        r"""添加 2-qubit 通用门，这个通用门需要 15 个参数。

        Args:
            theta (Tensor): 2-qubit 通用门的参数，其维度为 ``(15, )``
            which_qubits(list): 作用的量子比特编号
        """

        super(LoccAnsatz, self).universal_2_qubit_gate(theta, which_qubits)

    def universal_3_qubit_gate(self, theta, which_qubits):
        r"""添加 3-qubit 通用门，这个通用门需要 81 个参数。

        Note: 
            参考: https://cds.cern.ch/record/708846/files/0401178.pdf

        Args:
            theta (Tensor): 3-qubit 通用门的参数，其维度为 ``(81, )``
            which_qubits(list): 作用的量子比特编号
        """

        super(LoccAnsatz, self).universal_3_qubit_gate(theta, which_qubits)

    def superposition_layer(self):
        r"""添加一层 Hadamard 门。
        """
        for which_qubit in self.party.qubits:
            self.h(which_qubit)

    def weak_superposition_layer(self):
        r"""添加一层旋转角度为 :math:`\pi/4` 的 Ry 门。
        """
        _theta = paddle.to_tensor(np.array([np.pi / 4]))
        for which_qubit in self.party.qubits:
            self.ry(_theta, which_qubit)

    def linear_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 Ry 门，Rz 门和 CNOT 门的线性纠缠层。

        Attention:
            ``theta`` 的维度为 ``(depth, m, 2)`` ，最低维内容为对应的 ``ry`` 和 ``rz`` 的参数。

        Args:
            theta (Tensor): Ry 门和 Rz 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits(list): 作用的量子比特编号
        """
        assert self.m > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 2, 'the shape of theta is not right'
        # assert theta.shape[1] == self.m, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = list(range(self.m))

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.ry(theta[repeat][i][0], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            for i, q in enumerate(which_qubits):
                self.rz(theta[repeat][i][1], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i + 1], which_qubits[i]])

    def real_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 Ry 门和 CNOT 门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, m, 1)``

        Args:
            theta (Tensor): Ry 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits(list): 作用的量子比特编号
        """
        assert self.m > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 1, 'the shape of theta is not right'
        # assert theta.shape[1] == len(self.party), 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = list(range(self.m))

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.ry(theta[repeat][i][0], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            self.cnot([which_qubits[-1], which_qubits[0]])

    def complex_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 U3 门和 CNOT 门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为复数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, m, 3)``，最低维内容为对应的 ``u3`` 的参数 ``(theta, phi, lam)``

        Args:
            theta (Tensor): U3 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits(list): 作用的量子比特编号
        """
        assert self.m > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 3, 'the shape of theta is not right'
        # assert theta.shape[1] == self.m, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = list(range(self.m))

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.u3(theta[repeat][i][0], theta[repeat][i][1], theta[repeat][i][2], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            self.cnot([which_qubits[-1], which_qubits[0]])

    def real_block_layer(self, theta, depth):
        r"""添加 ``depth`` 层包含 Ry 门和 CNOT 门的弱纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。
        
        Attention:
            ``theta`` 的维度为 ``(depth, m-1, 4)`` 。
        
        Args:
            theta(Tensor): Ry 门的旋转角度
            depth(int): 纠缠层的深度
        """
        assert self.m > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'The dimension of theta is not right'
        _depth, _number, block = theta.shape
        assert depth > 0, 'depth must be greater than zero'
        assert _depth == depth, 'the depth of parameters has a mismatch'
        assert _number == self.m - 1 and block == 4, 'The shape of theta is not right'

        if self.m % 2 == 0:
            for i in range(depth):
                self.__add_real_layer(theta[i][:int(self.m / 2)], [0, self.m - 1])
                self.__add_real_layer(theta[i][int(self.m / 2):], [1, self.m - 2]) if self.m > 2 else None
        else:
            for i in range(depth):
                self.__add_real_layer(theta[i][:int((self.m - 1) / 2)], [0, self.m - 2])
                self.__add_real_layer(theta[i][int((self.m - 1) / 2):], [1, self.m - 1])

    def complex_block_layer(self, theta, depth):
        r"""添加 ``depth`` 层包含 U3 门和 CNOT 门的弱纠缠层。

        Note:
            这一层量子门的数学表示形式为复数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, m-1, 12)`` 。

        Args:
            theta (Tensor): U3 门的角度信息
            depth (int): 纠缠层的深度
        """
        assert self.m > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'The dimension of theta is not right'
        assert depth > 0, 'depth must be greater than zero'
        _depth, _number, block = theta.shape
        assert _depth == depth, 'the depth of parameters has a mismatch'
        assert _number == self.m - 1 and block == 12, 'The shape of theta is not right'

        if self.m % 2 == 0:
            for i in range(depth):
                self.__add_complex_layer(theta[i][:int(self.m / 2)], [0, self.m - 1])
                self.__add_complex_layer(theta[i][int(self.m / 2):], [1, self.m - 2]) if self.n > 2 else None
        else:
            for i in range(depth):
                self.__add_complex_layer(theta[i][:int((self.m - 1) / 2)], [0, self.m - 2])
                self.__add_complex_layer(theta[i][int((self.m - 1) / 2):], [1, self.m - 1])

    def __add_real_block(self, theta, position):
        r"""
        Add a real block to the circuit in (position). theta is a one dimensional tensor

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert len(theta) == 4, 'the length of theta is not right'
        assert 0 <= position[0] < self.m and 0 <= position[1] < self.m, 'position is out of range'
        self.ry(theta[0], position[0])
        self.ry(theta[1], position[1])

        self.cnot([position[0], position[1]])

        self.ry(theta[2], position[0])
        self.ry(theta[3], position[1])

    def __add_complex_block(self, theta, position):
        r"""
        Add a complex block to the circuit in (position). theta is a one dimensional tensor

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert len(theta) == 12, 'the length of theta is not right'
        assert 0 <= position[0] < self.m and 0 <= position[1] < self.m, 'position is out of range'
        self.u3(theta[0], theta[1], theta[2], position[0])
        self.u3(theta[3], theta[4], theta[5], position[1])

        self.cnot([position[0], position[1]])

        self.u3(theta[6], theta[7], theta[8], position[0])
        self.u3(theta[9], theta[10], theta[11], position[1])

    def __add_real_layer(self, theta, position):
        r"""
        Add a real layer on the circuit. theta is a two dimensional tensor. position is the qubit range the layer needs to cover

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert theta.shape[1] == 4 and theta.shape[0] == (position[1] - position[0] + 1) / 2, \
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_real_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def __add_complex_layer(self, theta, position):
        r"""
        Add a complex layer on the circuit.
        Theta is a two dimensional tensor.
        Position is the qubit range the layer needs to cover

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert theta.shape[1] == 12 and theta.shape[0] == (position[1] - position[0] + 1) / 2, \
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_complex_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def amplitude_damping(self, gamma, which_qubit):
        r"""添加振幅阻尼信道。

        其 Kraus 算符为：
        
        .. math::

            E_0 = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{bmatrix},
            E_1 = \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix}.

        Args:
            gamma (float): 减振概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).amplitude_damping(gamma, which_qubit)

    def generalized_amplitude_damping(self, gamma, p, which_qubit):
        r"""添加广义振幅阻尼信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt(p) \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{bmatrix},
            E_1 = \sqrt(p) \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
            E_2 = \sqrt(1-p) \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
            E_3 = \sqrt(1-p) \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

        Args:
            gamma (float): 减振概率，其值应该在 :math:`[0, 1]` 区间内
            p (float): 激发概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).generalized_amplitude_damping(gamma, p, which_qubit)

    def phase_damping(self, gamma, which_qubit):
        r"""添加相位阻尼信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{bmatrix},
            E_1 = \begin{bmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{bmatrix}.

        Args:
            gamma (float): phase damping 信道的参数，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).phase_damping(gamma, which_qubit)

    def bit_flip(self, p, which_qubit):
        r"""添加比特反转信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1-p} I,
            E_1 = \sqrt{p} X.

        Args:
            p (float): 发生 bit flip 的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).bit_flip(p, which_qubit)

    def phase_flip(self, p, which_qubit):
        r"""添加相位反转信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1 - p} I,
            E_1 = \sqrt{p} Z.

        Args:
            p (float): 发生 phase flip 的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).phase_flip(p, which_qubit)

    def bit_phase_flip(self, p, which_qubit):
        r"""添加比特相位反转信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1 - p} I,
            E_1 = \sqrt{p} Y.

        Args:
            p (float): 发生 bit phase flip 的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).bit_phase_flip(p, which_qubit)

    def depolarizing(self, p, which_qubit):
        r"""添加去极化信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1-p} I,
            E_1 = \sqrt{p/3} X,
            E_2 = \sqrt{p/3} Y,
            E_3 = \sqrt{p/3} Z.

        Args:
            p (float): depolarizing 信道的参数，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).depolarizing(p, which_qubit)

    def pauli_channel(self, p_x, p_y, p_z, which_qubit):
        r"""添加泡利信道。

        Args:
            p_x (float): 泡利矩阵 X 的对应概率，其值应该在 :math:`[0, 1]` 区间内
            p_y (float): 泡利矩阵 Y 的对应概率，其值应该在 :math:`[0, 1]` 区间内
            p_z (float): 泡利矩阵 Z 的对应概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数

        Note:
            三个输入的概率加起来需要小于等于 1。
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).pauli_channel(p_x, p_y, p_z, which_qubit)

    def customized_channel(self, ops, which_qubit):
        r"""添加自定义的量子信道。

        Args:
            ops (list): 表示信道的 Kraus 算符的列表
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, m)` 范围内， :math:`m` 为该参与方的量子比特数
        """
        which_qubit = self.party[which_qubit]
        super(LoccAnsatz, self).customized_channel(ops, which_qubit)


class LoccNet(paddle.nn.Layer):
    r"""用于设计我们的 LOCC 下的 protocol，并进行验证或者训练。
    """

    def __init__(self):
        r"""构造函数，用于实例化一个 LoccNet 对象。
        """
        super(LoccNet, self).__init__()
        self.parties_by_number = list()
        self.parties_by_name = dict()
        self.init_status = LoccStatus()

    def set_init_status(self, state, which_qubits):
        r"""对 LoccNet 的初始 LOCC 态节点进行初始化。

        Warning:
            该方法已弃用，请使用 ``set_init_state()`` 方法以代替。

        Args:
            state (Tensor): 输入的量子态的矩阵形式
            which_qubits (tuple or list): 该量子态所对应的量子比特，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``
        """
        warnings.warn('The member method set_init_status() is deprecated and please use set_init_state() instead.',
                      DeprecationWarning)

        self.set_init_state(state, which_qubits)

    def set_init_state(self, state, which_qubits):
        r"""对 LoccNet 的初始 LOCC 态节点进行初始化。

        Args:
            state (Tensor): 输入的量子态的矩阵形式
            which_qubits (tuple or list): 该量子态所对应的量子比特，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``
        """
        if isinstance(which_qubits, tuple):
            which_qubits = [which_qubits]
        temp_len = int(log2(sqrt(self.init_status.state.numpy().size)))
        self.init_status.state = kron(self.init_status.state, state)
        for idx, (party_id, qubit_id) in enumerate(which_qubits):
            if isinstance(party_id, str):
                self.parties_by_name[party_id][qubit_id] = (temp_len + idx)
            elif isinstance(party_id, int):
                self.parties_by_number[party_id][qubit_id] = (temp_len + idx)
            else:
                raise ValueError

    def partial_state(self, status, which_qubits, is_desired=True):
        r"""得到你想要的部分量子比特上的量子态。

        Args:
            status (LoccStatus or list): 输入的 LOCC 态节点，其类型应该为 ``LoccStatus`` 或者由其组成的 ``list``
            which_qubits (tuple or list): 指定的量子比特编号，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``
            is_desired (bool, optional): 默认是 ``True`` ，即返回量子比特上的部分量子态。如果为 ``False`` ，则抛弃这部分量子比特上的量子态，返回剩下的部分量子态

        Returns:
            LoccStatus or list: 得到部分量子态后的 LOCC 态节点，其类型为 ``LoccStatus`` 或者由其组成的 ``list``
        """
        if isinstance(which_qubits, tuple):
            which_qubits = [which_qubits]
        qubits_list = list()
        for party_id, qubit_id in which_qubits:
            if isinstance(party_id, str):
                qubits_list.append(self.parties_by_name[party_id][qubit_id])
            elif isinstance(party_id, int):
                qubits_list.append(self.parties_by_number[party_id][qubit_id])
            else:
                raise ValueError
        m = len(qubits_list)
        if isinstance(status, LoccStatus):
            n = int(log2(sqrt(status.state.numpy().size)))
        elif isinstance(status, list):
            n = int(log2(sqrt(status[0].state.numpy().size)))
        else:
            raise ValueError("can't recognize the input status")

        assert max(qubits_list) <= n, "qubit index out of range"
        origin_seq = list(range(0, n))
        target_seq = [idx for idx in origin_seq if idx not in qubits_list]
        target_seq = qubits_list + target_seq

        swaped = [False] * n
        swap_list = []
        for idx in range(0, n):
            if not swaped[idx]:
                next_idx = idx
                swaped[next_idx] = True
                while not swaped[target_seq[next_idx]]:
                    swaped[target_seq[next_idx]] = True
                    swap_list.append((next_idx, target_seq[next_idx]))
                    next_idx = target_seq[next_idx]

        cir = UAnsatz(n)
        for a, b in swap_list:
            cir.swap([a, b])

        if isinstance(status, LoccStatus):
            state = cir.run_density_matrix(status.state)
            if is_desired:
                state = partial_trace(state, 2 ** m, 2 ** (n - m), 2)
            else:
                state = partial_trace(state, 2 ** m, 2 ** (n - m), 1)
            new_status = LoccStatus(state, status.prob, status.measured_result)
        elif isinstance(status, list):
            new_status = list()
            for each_status in status:
                state = cir.run_density_matrix(each_status.state)
                if is_desired:
                    state = partial_trace(state, 2 ** m, 2 ** (n - m), 2)
                else:
                    state = partial_trace(state, 2 ** m, 2 ** (n - m), 1)
                new_status.append(LoccStatus(state, each_status.prob, each_status.measured_result))
        else:
            raise ValueError("can't recognize the input status")

        return new_status

    def reset_state(self, status, state, which_qubits):
        r"""用于重置你想要的量子比特上的部分量子态。

        Args:
            status (LoccStatus or list): 输入的 LOCC 态节点，其类型应该为 ``LoccStatus`` 或者由其组成的 ``list``
            state (Tensor): 输入的量子态的矩阵形式
            which_qubits (tuple or list): 指定需要被重置的量子比特编号，其形式为 ``(party_id, qubit_id)`` 的 ``tuple``，或者由其组成的 ``list``

        Returns:
            LoccStatus or list: 重置部分量子比特的量子态后的 LOCC 态节点，其类型为 ``LoccStatus`` 或者由其组成的 ``list``
        """
        if isinstance(which_qubits, tuple):
            which_qubits = [which_qubits]
        qubits_list = list()
        for party_id, qubit_id in which_qubits:
            if isinstance(party_id, str):
                qubits_list.append(self.parties_by_name[party_id][qubit_id])
            elif isinstance(party_id, int):
                qubits_list.append(self.parties_by_number[party_id][qubit_id])
            else:
                raise ValueError
        m = len(qubits_list)
        if isinstance(status, LoccStatus):
            n = int(log2(sqrt(status.state.numpy().size)))
        elif isinstance(status, list):
            n = int(log2(sqrt(status[0].state.numpy().size)))
        else:
            raise ValueError("can't recognize the input status")
        assert max(qubits_list) <= n, "qubit index out of range"

        origin_seq = list(range(0, n))
        target_seq = [idx for idx in origin_seq if idx not in qubits_list]
        target_seq = qubits_list + target_seq

        swaped = [False] * n
        swap_list = []
        for idx in range(0, n):
            if not swaped[idx]:
                next_idx = idx
                swaped[next_idx] = True
                while not swaped[target_seq[next_idx]]:
                    swaped[target_seq[next_idx]] = True
                    swap_list.append((next_idx, target_seq[next_idx]))
                    next_idx = target_seq[next_idx]

        cir0 = UAnsatz(n)
        for a, b in swap_list:
            cir0.swap([a, b])

        cir1 = UAnsatz(n)
        swap_list.reverse()
        for a, b in swap_list:
            cir1.swap([a, b])

        if isinstance(status, LoccStatus):
            _state = cir0.run_density_matrix(status.state)
            _state = partial_trace(_state, 2 ** m, 2 ** (n - m), 1)
            _state = kron(state, _state)
            _state = cir1.run_density_matrix(_state)
            new_status = LoccStatus(_state, status.prob, status.measured_result)
        elif isinstance(status, list):
            new_status = list()
            for each_status in status:
                _state = cir0.run_density_matrix(each_status.state)
                _state = partial_trace(_state, 2 ** m, 2 ** (n - m), 1)
                _state = kron(state, _state)
                _state = cir1.run_density_matrix(_state)
                new_status.append(LoccStatus(_state, each_status.prob, each_status.measured_result))
        else:
            raise ValueError("can't recognize the input status")

        return new_status

    def add_new_party(self, qubits_number, party_name=None):
        r"""添加一个新的 LOCC 的参与方。

        Args:
            qubits_number (int): 参与方的量子比特个数
            party_name (str, optional): 可选参数，默认为 ``None``，参与方的名字
        
        Note:
            你可以使用字符串或者数字对 party 进行索引。如果你想使用字符串索引，需要每次指定 ``party_name``；如果你想使用数字索引，则不需要指定 ``party_name``，其索引数字会从 0 开始依次增长。

        Returns:
            int or str: 该参与方的 ID，为数字或者字符串
        """
        party_id = None
        if party_name is None:
            party_id = party_name
            party_name = str(len(self.parties_by_name))
        elif isinstance(party_name, str) is False:
            raise ValueError
        if party_id is None:
            party_id = len(self.parties_by_name)

        new_party = LoccParty(qubits_number)
        self.parties_by_name[party_name] = new_party
        self.parties_by_number.append(new_party)
        
        return party_id

    def create_ansatz(self, party_id):
        r"""创建一个新的本地电路模板。

        Args:
            party_id (int or str): 参与方的 ID

        Returns:
            LoccAnsatz: 一个本地的量子电路
        """
        if isinstance(party_id, int):
            return LoccAnsatz(self.parties_by_number[party_id], self.get_qubit_number())
        elif isinstance(party_id, str):
            return LoccAnsatz(self.parties_by_name[party_id], self.get_qubit_number())
        else:
            raise ValueError

    def __measure_parameterized(self, state, which_qubits, result_desired, theta):
        r"""进行参数化的测量。

        Args:
            state (Tensor): 输入的量子态
            which_qubits (list): 测量作用的量子比特编号
            result_desired (str): 期望得到的测量结果
            theta (Tensor): 测量运算的参数

        Returns:
            Tensor: 测量坍塌后的量子态
            Tensor：测量坍塌得到的概率
            str: 测量得到的结果
        """
        n = self.get_qubit_number()
        assert len(which_qubits) == len(result_desired), \
            "the length of qubits wanted to be measured and the result desired should be same"
        op_list = [paddle.to_tensor(np.eye(2, dtype=np.complex128))] * n
        for idx in range(0, len(which_qubits)):
            i = which_qubits[idx]
            ele = result_desired[idx]
            if int(ele) == 0:
                basis0 = paddle.to_tensor(np.array([[1, 0], [0, 0]], dtype=np.complex128))
                basis1 = paddle.to_tensor(np.array([[0, 0], [0, 1]], dtype=np.complex128))
                rho0 = multiply(basis0, cos(theta[idx]))
                rho1 = multiply(basis1, sin(theta[idx]))
                rho = add(rho0, rho1)
                op_list[i] = rho
            elif int(ele) == 1:
                # rho = diag(concat([cos(theta[idx]), sin(theta[idx])]))
                # rho = paddle.to_tensor(rho, zeros((2, 2), dtype="float64"))
                basis0 = paddle.to_tensor(np.array([[1, 0], [0, 0]], dtype=np.complex128))
                basis1 = paddle.to_tensor(np.array([[0, 0], [0, 1]], dtype=np.complex128))
                rho0 = multiply(basis0, sin(theta[idx]))
                rho1 = multiply(basis1, cos(theta[idx]))
                rho = add(rho0, rho1)
                op_list[i] = rho
            else:
                print("cannot recognize the result_desired.")
            # rho = paddle.to_tensor(ones((2, 2), dtype="float64"), zeros((2, 2), dtype="float64"))
        measure_operator = paddle.to_tensor(op_list[0])
        if n > 1:
            for idx in range(1, len(op_list)):
                measure_operator = kron(measure_operator, op_list[idx])
        state_measured = matmul(matmul(measure_operator, state), dagger(measure_operator))
        prob = real(trace(matmul(matmul(dagger(measure_operator), measure_operator), state)))
        state_measured = divide(state_measured, prob)
        return state_measured, prob, result_desired

    def __measure_parameterless(self, state, which_qubits, result_desired):
        r"""进行 01 测量。

        Args:
            state (Tensor): 输入的量子态
            which_qubits (list): 测量作用的量子比特编号
            result_desired (str): 期望得到的测量结果

        Returns:
            Tensor: 测量坍塌后的量子态
            Tensor：测量坍塌得到的概率
            str: 测量得到的结果
        """
        n = self.get_qubit_number()
        assert len(which_qubits) == len(result_desired), \
            "the length of qubits wanted to be measured and the result desired should be same"
        op_list = [np.eye(2, dtype=np.complex128)] * n
        for i, ele in zip(which_qubits, result_desired):
            k = int(ele)
            rho = np.zeros((2, 2), dtype=np.complex128)
            rho[int(k), int(k)] = 1
            op_list[i] = rho
        if n > 1:
            measure_operator = paddle.to_tensor(NKron(*op_list))
        else:
            measure_operator = paddle.to_tensor(op_list[0])
        state_measured = matmul(matmul(measure_operator, state), dagger(measure_operator))
        prob = real(trace(matmul(matmul(dagger(measure_operator), measure_operator), state)))
        state_measured = divide(state_measured, prob)
        return state_measured, prob, result_desired

    def measure(self, status, which_qubits, results_desired, theta=None):
        r"""对 LOCC 态节点进行 01 测量或者含参测量。

        Args:
            status (LoccStatus or list): 输入的量子态，其类型应该为 ``LoccStatus`` 或者由其组成的 ``list``
            which_qubits (tuple or list): 测量作用的量子比特编号，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``
            results_desired (str or list): 期望得到的测量结果，用字符串进行表示，其类型为 ``str`` 或者由 ``str`` 组成的 ``list``，如 ``"0"``、``"1"`` 或者 ``["0", "1"]``
            theta (Tensor): 测量运算的参数，默认是 ``None``，表示 01 测量；若要使用含参测量则需要赋值

        Returns:
            LoccStatus or list: 测量后得到的 LOCC 态节点，其类型为 ``LoccStatus`` 或者由其组成的 ``list``
        """
        if isinstance(which_qubits, tuple):
            which_qubits = [which_qubits]
        if isinstance(results_desired, str):
            results_desired = [results_desired]
        elif not isinstance(results_desired, list):
            raise ValueError("cannot recognize the input of results_desired")

        qubits_list = list()
        for party_id, qubit_id in which_qubits:
            if isinstance(party_id, int):
                qubits_list.append((self.parties_by_number[party_id][qubit_id]))
            elif isinstance(party_id, str):
                qubits_list.append((self.parties_by_name[party_id][qubit_id]))
            else:
                raise ValueError

        if isinstance(status, LoccStatus):
            existing_result = status.measured_result
            prior_prob = status.prob
            new_status = list()
            for result_desired in results_desired:
                if theta is None:
                    result_measured = self.__measure_parameterless(status.state, qubits_list, result_desired)
                else:
                    result_measured = self.__measure_parameterized(status.state, qubits_list, result_desired, theta)
                state, prob, res = result_measured
                new_status.append(LoccStatus(state, prior_prob * prob, existing_result + res))
            if len(new_status) == 1:
                new_status = new_status[0]
        elif isinstance(status, list):
            new_status = list()
            for each_status in status:
                existing_result = each_status.measured_result
                prior_prob = each_status.prob
                for result_desired in results_desired:
                    if theta is None:
                        result_measured = self.__measure_parameterless(each_status.state, qubits_list, result_desired)
                    else:
                        result_measured = self.__measure_parameterized(each_status.state, qubits_list, result_desired, theta)
                    state, prob, res = result_measured
                    new_status.append(LoccStatus(state, prior_prob * prob, existing_result + res))
        else:
            raise ValueError("can't recognize the input status")

        return new_status

    def get_qubit_number(self):
        r"""得到该 LoccNet 的量子比特数量。

        Returns:
            int: 当前量子比特数量
        """
        qubits_number = 0
        for party in self.parties_by_number:
            qubits_number += party.qubit_number
        return qubits_number
