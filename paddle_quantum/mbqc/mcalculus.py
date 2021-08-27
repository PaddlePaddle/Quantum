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

"""
此模块包含处理 MBQC 测量模式的相关操作。
"""

from numpy import pi
from paddle import to_tensor, multiply
from paddle_quantum.mbqc.qobject import Pattern, Circuit
from paddle_quantum.mbqc.utils import div_str_to_float, int_to_div_str, print_progress

__all__ = [
    "MCalculus"
]


class MCalculus:
    r"""定义测量模式类。

    跟据文献 [The measurement calculus, arXiv: 0704.1263] 的测量语言，该类提供处理测量模式的各种基本操作。
    """

    def __init__(self):
        r"""``MCalculus`` 的构造函数，用于实例化一个 ``MCalculus`` 对象。
        """
        self.__circuit_slice = []  # Restore the information of sliced circuit
        self.__wild_pattern = []  # Record the background pattern
        self.__wild_commands = []  # Record wild commands information
        self.__pattern = None  # Record standard pattern
        self.__measured_qubits = []  # Record the measured qubits in the circuit model
        self.__circuit_width = None  # Record the circuit width
        self.__track = False  # Switch of progress bar

    def track_progress(self, track=True):
        r"""显示测量模式处理过程的进度条开关。

        Args:
            track (bool, optional): ``True`` 为打开进度条显示，``False`` 为关闭进度条显示，默认为 ``True``
        """
        assert isinstance(track, bool), "parameter 'track' must be a bool."
        self.__track = track

    def set_circuit(self, circuit):
        r"""对 ``MCalculus`` 类设置量子电路。

        Args:
            circuit (Circuit): 量子电路
        """
        assert isinstance(circuit, Circuit), "please input a parameter of type 'Circuit'."
        assert circuit.is_valid(), "the circuit is not valid as at least one qubit is not performed any gate yet."
        self.__circuit_width = circuit.get_width()
        self.__slice_circuit(circuit.get_circuit())
        for gate in self.__circuit_slice:
            self.__to_pattern(gate)
        self.__join_patterns()

    def __slice_circuit(self, circuit):
        r"""对电路进行切片操作，标记每个量子门和量子测量的输入比特和输出比特。

        Note:
            这是内部方法，用户不需要直接调用到该方法。
            在此，我们使用 ``str`` 类型的变量作为节点的标签。为了不丢失原先节点的坐标信息，便于后续的画图操作，
            我们使用形如 ``("1/1", "2/1")`` 类型作为所有节点的标签。

        Args:
            circuit (list): 电路列表，列表中每个元素代表一个量子门或测量
        """
        # Slice the circuit to mark the input_/output_ labels for measurement pattern
        counter = []
        for gate in circuit:
            name = gate[0]
            which_qubit = gate[1]
            assert len(which_qubit) in [1, 2], str(len(which_qubit)) + "-qubit gate is not supported in this version."

            if len(which_qubit) == 1:  # Single-qubit gates
                which_qubit = which_qubit[0]  # Take the item only
                assert which_qubit not in self.__measured_qubits, \
                    "please check your qubit index as this qubit has already been measured."
                input_ = [(int_to_div_str(which_qubit), int_to_div_str(int(counter.count(which_qubit))))]
                if name == 'm':
                    output_ = []  # No output_ node for measurement
                else:
                    output_ = [(int_to_div_str(which_qubit), int_to_div_str(int(counter.count(which_qubit) + 1)))]
                    counter += [which_qubit]  # Record the index
                # The gate after slicing has a form of:
                # [original_gate, input_, output_], e.g. = [[h, [0], None], input_, output_]
                self.__circuit_slice.append([gate, input_, output_])

            else:  # Two-qubit gates
                control = which_qubit[0]
                target = which_qubit[1]
                assert control not in self.__measured_qubits and target not in self.__measured_qubits, \
                    "please check your qubit indices as these qubits have already been measured."
                if name == 'cz':  # Input and output nodes coincide for CZ gate
                    input_output = [(int_to_div_str(control), int_to_div_str(int(counter.count(control)))),
                                    (int_to_div_str(target), int_to_div_str(int(counter.count(target))))]
                    # The gate after slicing has a form of:
                    # [original_gate, input_, output_], e.g. = [[cz, [0, 1], None], input_, output_]
                    self.__circuit_slice.append([gate, input_output, input_output])

                elif name == 'cnot':
                    input_ = [(int_to_div_str(control), int_to_div_str(int(counter.count(control)))),
                              (int_to_div_str(target), int_to_div_str(int(counter.count(target))))]
                    output_ = [(int_to_div_str(control), int_to_div_str(int(counter.count(control)))),
                               (int_to_div_str(target), int_to_div_str(int(counter.count(target) + 1)))]
                    counter += [target]  # Record the index
                    # The gate after slicing has a form of:
                    # [original_gate, input_, output_], e.g. = [[cnot, [0, 1], None], input_, output_]
                    self.__circuit_slice.append([gate, input_, output_])

                else:
                    input_ = [(int_to_div_str(control), int_to_div_str(int(counter.count(control)))),
                              (int_to_div_str(target), int_to_div_str(int(counter.count(target))))]
                    output_ = [(int_to_div_str(control), int_to_div_str(int(counter.count(control) + 1))),
                               (int_to_div_str(target), int_to_div_str(int(counter.count(target) + 1)))]
                    counter += which_qubit  # Record the index
                    # The gate after slicing has a form of:
                    # [original_gate, input_, output_], e.g. = [[h, [0], None], input_, output_]
                    self.__circuit_slice.append([gate, input_, output_])

    @staticmethod
    def __set_ancilla_label(input_, output_, ancilla_num_list=None):
        r"""插入辅助比特。

        在输入比特和输出比特中间插入辅助比特，辅助比特节点的坐标根据数目而均分，其标签类型与输入和输出比特的标签类型相同。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            input_ (list): 测量模式的输入节点
            output_ (list): 测量模式的输出节点
            ancilla_num_list (list): 需要插入的辅助节点个数列表

        Returns:
            list: 辅助节点标签列表
        """
        assert len(input_) == len(output_), "input and output must have same length."
        assert len(input_) in [1, 2], str(len(input_)) + "-qubit gate is not supported in this version."
        ancilla_num = [] if ancilla_num_list is None else ancilla_num_list
        ancilla_labels = []

        for i in range(len(ancilla_num)):
            input_qubit = input_[i]  # Obtain input qubit
            row_in = div_str_to_float(input_qubit[0])  # Row of input qubit
            col_in = div_str_to_float(input_qubit[1])  # Column of input qubit

            output_qubit = output_[i]  # Obtain output qubit
            row_out = div_str_to_float(output_qubit[0])  # Row of output qubit
            col_out = div_str_to_float(output_qubit[1])  # Column of output qubit

            assert row_in == row_out, "please check the qubit labels of your input."

            # Calculate Auxiliary qubits' positions
            col = col_out - col_in
            pos = [int_to_div_str(int(col_in * (ancilla_num[i] + 1) + j * col), ancilla_num[i] + 1)
                   for j in range(1, ancilla_num[i] + 1)]

            # Get the ancilla_labels
            for k in range(ancilla_num[i]):
                ancilla_labels.append((input_qubit[0], pos[k]))

        return ancilla_labels

    def __to_pattern(self, gate):
        r"""将量子电路中的门和测量翻译为等价的测量模式。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Warning:
            当前版本支持的量子门为 ``[H, X, Y, Z, S, T, Rx, Ry, Rz, Rz_5, U, CNOT, CNOT_15, CZ]`` 和单比特测量。
            注意量子门和测量对应的测量模式不唯一，本方法目前仅选取常用的一种或者两种测量模式进行翻译。

        Args:
            gate (list): 待翻译的量子门或量子测量，列表中存储的是原始量子门（其中包含量子门名称、作用比特、参数）、输入比特、输出比特
        """
        original_gate, input_, output_ = gate
        name, which_qubit, param = original_gate

        ancilla = []
        zero = to_tensor([0], dtype="float64")
        minus_one = to_tensor([-1], dtype="float64")
        half_pi = to_tensor([pi / 2], dtype="float64")
        minus_half_pi = to_tensor([-pi / 2], dtype="float64")
        minus_pi = to_tensor([-pi], dtype="float64")

        if name == 'h':  # Hadamard gate
            E = Pattern.CommandE([input_[0], output_[0]])
            M = Pattern.CommandM(input_[0], zero, "XY", [], [])
            X = Pattern.CommandX(output_[0], [input_[0]])
            commands = [E, M, X]

        elif name == 'x':  # Pauli X gate
            ancilla = self.__set_ancilla_label(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], zero, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], minus_pi, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == 'y':  # Pauli Y gate
            ancilla = self.__set_ancilla_label(input_, output_, [3])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], half_pi, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], half_pi, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[1], minus_half_pi, "XY", [], [input_[0], ancilla[0]])
            M4 = Pattern.CommandM(ancilla[2], zero, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == 'z':  # Pauli Z gate
            ancilla = self.__set_ancilla_label(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], minus_pi, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], zero, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == 's':  # Phase gate
            ancilla = self.__set_ancilla_label(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], minus_half_pi, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], zero, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == 't':  # T gate
            ancilla = self.__set_ancilla_label(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], to_tensor([-pi / 4], dtype="float64"), "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], zero, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == 'rx':  # Rotation gate around x axis
            ancilla = self.__set_ancilla_label(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], zero, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], multiply(param, minus_one), "XY", [input_[0]], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == 'ry':  # Rotation gate around y axis
            ancilla = self.__set_ancilla_label(input_, output_, [3])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], half_pi, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], multiply(param, minus_one), "XY", [input_[0]], [])
            M3 = Pattern.CommandM(ancilla[1], minus_half_pi, "XY", [], [input_[0], ancilla[0]])
            M4 = Pattern.CommandM(ancilla[2], zero, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == 'rz':  # Rotation gate around z axis
            ancilla = self.__set_ancilla_label(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], multiply(param, minus_one), "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], zero, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == 'rz_5':  # Rotation gate around z axis
            ancilla = self.__set_ancilla_label(input_, output_, [3])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], zero, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], zero, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[1], multiply(param, minus_one), "XY", [ancilla[0]], [input_[0]])
            M4 = Pattern.CommandM(ancilla[2], zero, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == 'u':  # General single-qubit unitary
            ancilla = self.__set_ancilla_label(input_, output_, [3])
            alpha, theta, gamma = param
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], multiply(alpha, minus_one), "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], multiply(theta, minus_one), "XY", [input_[0]], [])
            M3 = Pattern.CommandM(ancilla[1], multiply(gamma, minus_one), "XY", [ancilla[0]], [input_[0]])
            M4 = Pattern.CommandM(ancilla[2], zero, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == 'cnot':  # Control NOT gate
            ancilla = self.__set_ancilla_label(input_, output_, [0, 1])
            E23 = Pattern.CommandE([input_[1], ancilla[0]])
            E13 = Pattern.CommandE([input_[0], ancilla[0]])
            E34 = Pattern.CommandE([ancilla[0], output_[1]])
            M2 = Pattern.CommandM(input_[1], zero, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[0], zero, "XY", [], [])
            X4 = Pattern.CommandX(output_[1], [ancilla[0]])
            Z1 = Pattern.CommandZ(output_[0], [input_[1]])
            Z4 = Pattern.CommandZ(output_[1], [input_[1]])
            commands = [E23, E13, E34, M2, M3, X4, Z1, Z4]

        # Measurement pattern of CNOT by 15 qubits, c.f. [arXiv: quant-ph/0301052v2]
        # Note: due to the '1' in byproduct Z of qubit-7, we manually add a Z gate after qubit 7 to match this
        elif name == "cnot_15":  # Controlled Not gate
            input1, input2 = input_
            output1, output2 = output_
            ancilla = self.__set_ancilla_label(input_, output_, [7, 5])
            new_row = str(int(div_str_to_float(input2[0]) + div_str_to_float(input1[0]))) + "/" + "2"
            new_col_1 = (div_str_to_float(output1[1]) + div_str_to_float(input1[1]))
            new_col_2 = (div_str_to_float(output2[1]) + div_str_to_float(input2[1]))
            new_col = str(int((new_col_1 + new_col_2) / 2)) + "/" + "2"
            ancilla.append((new_row, new_col))
            E12 = Pattern.CommandE([input1, ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], ancilla[3]])
            E48 = Pattern.CommandE([ancilla[2], ancilla[12]])
            E56 = Pattern.CommandE([ancilla[3], ancilla[4]])
            E67 = Pattern.CommandE([ancilla[4], ancilla[5]])
            E716 = Pattern.CommandE([ancilla[5], ancilla[6]])
            E1617 = Pattern.CommandE([ancilla[6], output1])
            E910 = Pattern.CommandE([input2, ancilla[7]])
            E1011 = Pattern.CommandE([ancilla[7], ancilla[8]])
            E1112 = Pattern.CommandE([ancilla[8], ancilla[9]])
            E812 = Pattern.CommandE([ancilla[12], ancilla[9]])
            E1213 = Pattern.CommandE([ancilla[9], ancilla[10]])
            E1314 = Pattern.CommandE([ancilla[10], ancilla[11]])
            E1415 = Pattern.CommandE([ancilla[11], output2])
            M1 = Pattern.CommandM(input1, zero, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], half_pi, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[1], half_pi, "XY", [], [])
            M4 = Pattern.CommandM(ancilla[2], half_pi, "XY", [], [])
            M5 = Pattern.CommandM(ancilla[3], half_pi, "XY", [], [])
            M6 = Pattern.CommandM(ancilla[4], half_pi, "XY", [], [])
            M8 = Pattern.CommandM(ancilla[12], half_pi, "XY", [], [])
            M12 = Pattern.CommandM(ancilla[9], half_pi, "XY", [], [])
            M9 = Pattern.CommandM(input2, zero, "XY", [], [])
            M10 = Pattern.CommandM(ancilla[7], zero, "XY", [], [])
            M11 = Pattern.CommandM(ancilla[8], zero, "XY", [], [])
            M13 = Pattern.CommandM(ancilla[10], zero, "XY", [], [])
            M14 = Pattern.CommandM(ancilla[11], zero, "XY", [], [])
            M7 = Pattern.CommandM(ancilla[5], minus_pi, "XY", [], input_ + [ancilla[i] for i in [1, 2, 3, 8, 12]])
            M16 = Pattern.CommandM(ancilla[6], zero, "XY", [], [ancilla[i] for i in [0, 1, 3, 4]])
            X15 = Pattern.CommandX(output2, [ancilla[i] for i in [0, 1, 7, 9, 11, 12]])
            X17 = Pattern.CommandX(output1, [ancilla[6]])
            Z15 = Pattern.CommandZ(output2, [input2, ancilla[8], ancilla[10]])
            Z17 = Pattern.CommandZ(output1, [ancilla[5]])
            commands = [E12, E23, E34, E45, E48, E56, E67, E716, E1617, E910, E1011, E1112, E812, E1213, E1314, E1415,
                        M1, M2, M3, M4, M5, M6, M8, M12, M9, M10, M11, M13, M14, M7, M16, X15, X17, Z15, Z17]

        elif name == 'cz':  # Controlled Z gate
            commands = [Pattern.CommandE(input_)]

        elif name == 'm':  # Single-qubit measurement
            self.__measured_qubits.append(which_qubit[0])
            commands = [Pattern.CommandM(input_[0], param[0], param[1], param[2], param[3])]

        else:
            raise KeyError("translation of such gate is not supported in this version")

        self.__wild_commands += commands
        self.__wild_pattern.append(
            Pattern(str([name] + [str(qubit) for qubit in which_qubit]), list(set(input_ + output_ + ancilla)),
                    input_, output_, commands))

    def __list_to_net(self, bit_lst):
        r"""把一个包含比特标签的一维列表处理为二维网状结构的列表，与电路图的网状结构相对应。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            bit_lst (list): 输入或输出节点标签的列表

        Returns:
            list: 网状结构的二维列表，每行记录的是输入或输出节点所在的列数
        """
        bit_net = [[] for _ in range(self.__circuit_width)]
        for qubit in bit_lst:
            bit_net[int(div_str_to_float(qubit[0]))].append(div_str_to_float(qubit[1]))
        return bit_net

    def __get_input(self, bit_lst):
        r"""获得列表中的输入比特节点标签。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            bit_lst (list): 节点标签的列表

        Returns:
            list: 输入比特的节点标签列表
        """
        bit_net = self.__list_to_net(bit_lst)
        input_ = []
        for i in range(self.__circuit_width):
            input_.append((int_to_div_str(i), int_to_div_str(int(min(bit_net[i])))))
        return input_

    def __get_output(self, bit_lst):
        r"""获得列表中的输出比特节点标签。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            bit_lst (list): 节点标签的列表

        Returns:
            list: 输出比特的节点标签列表
        """
        bit_net = self.__list_to_net(bit_lst)
        # Need to separate the classical output and quantum output
        c_output = []
        q_output = []
        for i in range(self.__circuit_width):
            if i not in self.__measured_qubits:
                q_output.append((int_to_div_str(i), int_to_div_str(int(max(bit_net[i])))))
            else:
                c_output.append((int_to_div_str(i), int_to_div_str(int(max(bit_net[i])))))
        output_ = [c_output, q_output]
        return output_

    def __join_patterns(self):
        r"""将逐个翻译好的量子门和测量的测量模式拼接到一起。

        Note:
            这是内部方法，用户不需要直接调用到该方法。
        """
        names = ""
        input_list = []
        output_list = []
        space = []
        commands = []

        for pat in self.__wild_pattern:
            names += pat.name
            space += [qubit for qubit in pat.space if qubit not in space]
            input_list += pat.input_
            output_list += pat.output_
            commands += pat.commands
        input_ = self.__get_input(input_list)
        output_ = self.__get_output(output_list)
        self.__pattern = Pattern(names, space, input_, output_, commands)

    @staticmethod
    def __propagate(which_cmds):
        r"""交换两个命令。

        任意两个命令交换遵从对应算符的交换关系，详细的交换关系请参见 [arXiv:0704.1263]。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Hint:
            我们默认算符由左至右的顺序运算，例如 ``"E"`` 要在所有运算之前，因此 ``"E"`` 要在命令列表中的最左侧，其他命令以此类推。

        Warning:
            该方法中只对 ``"E"``, ``"M"``, ``"X"``, ``"Z"``, ``"S"`` 五类命令中的任意两个进行交换。

        Args:
            which_cmds (list): 待交换的命令列表

        Returns:
            list: 交换后的新命令列表
        """
        cmd1 = which_cmds[0]
        cmd2 = which_cmds[1]
        name1 = cmd1.name
        name2 = cmd2.name

        assert {name1, name2}.issubset(["E", "M", "X", "Z", "S"]), \
            "command's name must be in ['E', 'M', 'X', 'Z', 'S']."

        # [X, E] --> [E, X]
        if name1 == "X" and name2 == "E":
            X_qubit = cmd1.which_qubit
            E_qubits = cmd2.which_qubits[:]
            if X_qubit not in E_qubits:
                return [cmd2, cmd1]  # Independent commands commute
            else:  # For dependent commands
                op_qubit = list(set(E_qubits).difference([X_qubit]))
                new_cmd = Pattern.CommandZ(op_qubit[0], cmd1.domain)
                return [cmd2, new_cmd, cmd1]

        # [Z, E] --> [E, Z]
        elif name1 == "Z" and name2 == "E":
            return [cmd2, cmd1]  # they commute

        # [M, E] --> [E, M]
        elif name1 == "M" and name2 == "E":
            if cmd1.which_qubit not in cmd2.which_qubits:
                return [cmd2, cmd1]  # Independent commands commute
            else:
                raise ValueError("measurement command should be after entanglement command.")

        # [X, M] --> [M, X]
        elif name1 == "X" and name2 == "M":
            X_qubit = cmd1.which_qubit
            M_qubit = cmd2.which_qubit
            if X_qubit != M_qubit:
                return [cmd2, cmd1]  # Independent commands commute
            else:  # For dependent commands
                measurement_plane = cmd2.plane
                if measurement_plane == 'XY':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "XY", cmd2.domain_s + cmd1.domain, cmd2.domain_t)
                elif measurement_plane == 'YZ':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "YZ", cmd2.domain_s, cmd2.domain_t + cmd1.domain)
                else:
                    raise ValueError("in this version we only support measurements in the XY or YZ plane.")
                return [M_new]

        # [Z, M] --> [M, Z]
        elif name1 == "Z" and name2 == "M":
            Z_qubit = cmd1.which_qubit
            M_qubit = cmd2.which_qubit
            if Z_qubit != M_qubit:
                return [cmd2, cmd1]  # Independent commands commute
            else:  # For dependent commands
                measurement_plane = cmd2.plane
                if measurement_plane == 'YZ':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "YZ", cmd2.domain_s + cmd1.domain, cmd2.domain_t)
                elif measurement_plane == 'XY':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "XY", cmd2.domain_s, cmd2.domain_t + cmd1.domain)
                else:
                    raise ValueError("in this version we only support measurements in the XY or YZ plane.")
                return [M_new]

        # [Z, X] --> [X, Z]
        elif name1 == "Z" and name2 == "X":
            return [cmd2, cmd1]  # They commute

        # Merge two CommandX
        elif name1 == "X" and name2 == "X":
            X1_qubit = cmd1.which_qubit
            X2_qubit = cmd2.which_qubit
            if X1_qubit == X2_qubit:
                return [Pattern.CommandX(X1_qubit, cmd1.domain + cmd2.domain)]
            else:
                return which_cmds

        # Merge two CommandZ
        elif name1 == "Z" and name2 == "Z":
            Z1_qubit = cmd1.which_qubit
            Z2_qubit = cmd2.which_qubit
            if Z1_qubit == Z2_qubit:
                return [Pattern.CommandZ(Z1_qubit, cmd1.domain + cmd2.domain)]
            else:
                return which_cmds

        # [S, M or X or Z or S] -> [M or X or Z or S, S]
        elif name1 == "S":
            # The propagation rule of S with XY or YZ plane measurement is the same
            # [S, M] --> [M, S]
            if name2 == "M":
                S_qubit = cmd1.which_qubit
                S_domains = cmd1.domain
                # According to the reference [arXiv:0704.1263],
                # if S_qubit is in measurement command's domain_s or domain_t,
                # we need to add S's domain to the CommandM's domain_s or domain_t
                if S_qubit in cmd2.domain_s:
                    cmd2.domain_s += S_domains
                if S_qubit in cmd2.domain_t:
                    cmd2.domain_t += S_domains
                # If S_qubit is not in measurement command's domains, they can swap without modification
                return [cmd2, cmd1]
            # [S, X] --> [X, S], [S, Z] --> [Z, S], [S, S] --> [S, S]
            elif name2 == "X" or name2 == "Z" or name2 == "S":
                S_qubit = cmd1.which_qubit
                S_domains = cmd1.domain
                if S_qubit in cmd2.domain:
                    cmd2.domain += S_domains
                return [cmd2, cmd1]
            else:
                return which_cmds

        # Otherwise, keep the input commands unchanged
        else:
            return which_cmds

    def __propagate_by_type(self, cmd_type, cmds):
        r"""把列表中某指定类型的所有命令向前交换。

        向前交换不是指将其交换至整个命令列表的最左端，而是交换至规则允许范围内的最前方。
        例如：对同一比特的测量操作必须在纠缠操作的后面，所以当调用该方法交换测量算符时，测量算符被交换到了它前面的纠缠算符之后。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            cmd_type (str): 待交换的命令类型，为 ``"E"``, ``"M"``, ``"X"``, ``"Z"`` 或 ``"S"``
            cmds (list): 待处理的命令列表

        Returns:
            list: 交换后的新命令列表
        """
        assert cmd_type in ["E", "M", "X", "Z", "S"], "command's name must be 'E', 'M', 'X', 'Z' or 'S'."

        # Propagate commands 'E', 'M', 'X', 'Z' from back to front
        if cmd_type in ["E", "M", "X", "Z"]:
            for i in range(len(cmds) - 1, 0, -1):
                if cmds[i].name == cmd_type:
                    cmds = cmds[:i - 1] + self.__propagate([cmds[i - 1], cmds[i]]) + cmds[i + 1:]

        # Propagate commands 'S' from front to back
        elif cmd_type == "S":
            for i in range(0, len(cmds) - 1):
                if cmds[i].name == cmd_type:
                    cmds = cmds[:i] + self.__propagate([cmds[i], cmds[i + 1]]) + cmds[i + 2:]

        return cmds

    @staticmethod
    def __reorder_labels_by_row(labels):
        r"""将标签按照行数从小到大进行排序。

        该方法是为了调整输入节点和输出节点的顺序，从而与电路图的顺序相一致，便于后续运行和使用。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            labels (list): 待改写的标签列表

        Returns:
            list: 改写后的标签列表，按照行数从小到大顺序排列
        """
        row = []
        row_and_col = {}

        for label in labels:  # Obtain row and column index
            row.append(int(div_str_to_float(label[0])))
            row_and_col[int(div_str_to_float(label[0]))] = label[1]

        row.sort()  # Reorder the labels
        labels_in_order = [(int_to_div_str(i), row_and_col[i]) for i in row]

        return labels_in_order

    @staticmethod
    def __commands_to_numbers(cmds):
        r"""将命令列表映射成数字列表。

        映射规则为 CommandE -> 1, CommandM -> 2, CommandX -> 3, CommandZ -> 4, CommandS -> 5。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            cmds (list): 待处理的命令列表

        Returns:
            list: 记录每种命令的个数
            list: 映射后的列表
            list: 映射后的列表进行从小到大排序得到的标准列表
        """
        cmd_map = {"E": 1, "M": 2, "X": 3, "Z": 4, "S": 5}
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_standard = cmd_num_wild[:]
        cmd_num_standard.sort(reverse=False)
        cmds_count = [cmd_num_standard.count(i) for i in [1, 2, 3, 4, 5]]  # Count each type of commands

        return cmds_count, cmd_num_wild, cmd_num_standard

    def __distance_to_standard(self, cmds):
        r"""采用 Hamming 距离定义当前命令列表的顺序和标准顺序的距离函数。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            cmds (list): 当前命令列表
        """
        _, cmd_wild, cmd_std = self.__commands_to_numbers(cmds[:])

        return sum([cmd_wild[i] == cmd_std[i] for i in range(len(cmd_wild))]) / len(cmd_wild)

    def __is_standard(self, cmd_type, cmds):
        r"""判断命令列表中指定的命令类型是否为标准顺序。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            cmd_type (str): 待判断的命令名称，为 ``E``, ``M``, ``X``, ``Z`` 或 ``S``
            cmds (list): 待判断的命令列表

        Returns:
            bool: 列表是否为标准列表的布尔值
        """
        assert cmd_type in ["E", "M", "X", "Z", "S"], "command's name must be 'E', 'M', 'X', 'Z', or 'S'."

        # Map the commands to numbers
        cmds_count, cmd_num_wild, cmd_num_standard = self.__commands_to_numbers(cmds)
        pointer_map = {"E": sum(cmds_count[:1]),  # Number of commands E
                       "M": sum(cmds_count[:2]),  # Number of commands E + M
                       "X": sum(cmds_count[:3]),  # Number of commands E + M + X
                       "Z": sum(cmds_count[:4]),  # Number of commands E + M + X + Z
                       "S": sum(cmds_count[:5])}  # Number of commands E + M + X + Z + S

        return cmd_num_wild[:pointer_map[cmd_type]] == cmd_num_standard[:pointer_map[cmd_type]]

    def __simplify_pauli_measurements(self):
        r"""对节点依赖性进行简化。

        在某些特殊情形下，测量节点对其他节点的依赖性可以被简化。
        设 \alpha 为不考虑依赖关系的测量角度，则测量角度实际上只有四种可能性，分别为：

         .. math::

            \theta_{\text{ad}} = \alpha

            \theta_{\text{ad}} = \alpha + \pi

            \theta_{\text{ad}} = - \alpha

            \theta_{\text{ad}} = - \alpha + \pi

            \text{当 } \alpha \text{ 为 } 0, \pi / 2, \pi, 3 \times \pi / 2 \text{ 时，该依赖关系可以简化。}

            \text{例如 } \alpha = \pi \text{ 时，}\pm \alpha + t \times \pi \text{导致的测量效果一样，与 domain\_s 无关，因此 domain\_s 可以移除。其他情形同理。}

        Note:
        这是内部方法，用户不需要直接调用到该方法。
        """
        for cmd in self.__pattern.commands:
            if cmd.name == 'M':  # Find CommandM
                remainder = cmd.angle.numpy().item() % (2 * pi)
                if remainder in [0, pi]:
                    cmd.domain_s = []
                elif remainder in [pi / 2, (3 * pi) / 2]:
                    cmd.domain_t += cmd.domain_s[:]
                    cmd.domain_s = []

    def standardize(self):
        r"""对测量模式进行标准化。

        该方法对测量模式进行标准化操作，转化成等价的 EMC 模型。即将所有的 ``CommandE`` 交换到最前面，其次是 ``CommandM``，
        ``CommandX`` 和 ``CommandZ``。为了简化测量模式，该方法在标准化各类命令之后还对 ``CommandM`` 进行 Pauli 简化。
        """
        cmds = self.__pattern.commands

        for cmd_type in ["E", "M", "X", "Z"]:
            while not self.__is_standard(cmd_type, cmds):
                cmds = self.__propagate_by_type(cmd_type, cmds)
                print_progress(self.__distance_to_standard(cmds), "Standardization Progress", self.__track)

        self.__pattern.commands = cmds
        self.__simplify_pauli_measurements()

    @staticmethod
    def __pull_out_domain_t(cmds):
        r"""在命令列表中把信号转移算符从测量算符中提取出来。

        Note:
            信号转移是一种特殊的操作，通过与其他命令算符的交换，解除测量命令的域 t 列表中节点的依赖性，
            从而在某些情况下简化测量模式，详情请参见 [arXiv:0704.1263]。

        Warning:
            我们只提取 XY 平面测量算符的域 t 列表中节点的依赖性作为信号转移算符。对于 YZ 平面测量算符，我们并未采取此操作。

        Args:
            cmds (list): 命令列表

        Returns:
            list: 提取信号转移算符后的命令列表
        """
        cmds_len = len(cmds)
        for i in range(cmds_len - 1, -1, -1):
            cmd = cmds[i]
            if cmd.name == "M" and cmd.plane == 'XY':
                signal_cmd = Pattern.CommandS(cmd.which_qubit, cmd.domain_t)
                cmd.domain_t = []
                cmds = cmds[:i] + [cmd, signal_cmd] + cmds[i + 1:]
        return cmds

    def shift_signals(self):
        r"""信号转移操作。

        Note:
            这是用户选择性调用的方法之一。
        """
        cmds = self.__pattern.commands
        cmds = self.__pull_out_domain_t(cmds)

        # Propagate CommandS
        while not self.__is_standard("S", cmds):
            cmds = self.__propagate_by_type("S", cmds)
            print_progress(self.__distance_to_standard(cmds), "Signal Shifting Progress", self.__track)

        # Kick out all the CommandS in the cmd list
        cmds = [cmd for cmd in cmds if cmd.name != "S"]
        self.__pattern.commands = cmds

    def get_pattern(self):
        r"""返回测量模式。

        Returns:
            Pattern: 处理后的测量模式
        """
        return self.__pattern

    @staticmethod
    def __default_order_by_row(labels):
        r"""按照行的顺序对节点标签列表进行排序并返回。

        排序规则：不同行数，行数小的优先。相同行数，列数小的优先。

        Note:
            这是内部方法，用户不需要直接调用到该方法。

        Args:
            labels (list): 待处理的节点标签列表

        Returns:
            list: 排序后的节点标签列表
        """
        # Construct a dict by string labels and their float values
        labels_dict = {label: (div_str_to_float(label[0]), div_str_to_float(label[1])) for label in labels}
        # Sort the dict by values (sort row first and then column)
        sorted_dict = dict(sorted(labels_dict.items(), key=lambda item: item[1]))
        # Extract the keys in the dict
        labels_sorted = list(sorted_dict.keys())

        return labels_sorted

    def optimize_by_row(self):
        r"""按照行序优先的原则对测量模式中的测量顺序进行优化。

        Warning:
            这是一种启发式的优化算法，对于特定的测量模式可以起到优化测量顺序的作用，不排除存在更优的测量顺序。例如，对于浅层量子电路，
            按照行序优先原则，测量完同一量子位上的量子门、测量对应的节点后，该量子位不再起作用，进而减少后续计算时可能涉及到的节点数目。
        """
        cmds = self.__pattern.commands

        # Split the commands by type
        cmdE_list = [cmd for cmd in cmds if cmd.name == "E"]
        cmdM_list = [cmd for cmd in cmds if cmd.name == "M"]
        cmdC_list = [cmd for cmd in cmds if cmd.name in ["X", "Z"]]
        # Construct a dict from qubit labels and their measurement commands
        cmdM_map = {cmd.which_qubit: cmd for cmd in cmdM_list}

        # Sort all the qubit labels by row
        cmdM_qubit_list = self.__default_order_by_row([cmdM.which_qubit for cmdM in cmdM_list])
        mea_length = len(cmdM_qubit_list)

        for i in range(mea_length):
            optimal = False
            while not optimal:  # If the qubits list is not standard
                # Slice measurement qubit list into three parts
                measured = cmdM_qubit_list[:i]
                measuring = cmdM_qubit_list[i]
                to_measure = cmdM_qubit_list[i + 1:]

                domains = set(cmdM_map[measuring].domain_s + cmdM_map[measuring].domain_t)
                # Find the qubits in domain but not in front of the current measurement
                push = self.__default_order_by_row(list(domains.difference(measured)))
                if push:  # Remove qubits from the to_measure list and push it to the front
                    to_measure = [qubit for qubit in to_measure if qubit not in push]
                    cmdM_qubit_list = cmdM_qubit_list[:i] + push + [measuring] + to_measure
                else:  # If no push qubits then jump out of the while loop
                    optimal = True
            print_progress((i + 1) / mea_length, "Measurement Opt. Progress", self.__track)

        # Sort the measurement commands by the sorted qubit labels
        cmdM_opt = [cmdM_map[which_qubit] for which_qubit in cmdM_qubit_list]

        # Update pattern
        cmds = cmdE_list + cmdM_opt + cmdC_list
        self.__pattern.commands = cmds
