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

r"""
This module contains the related operations for the MBQC measurement patterns.
"""

from numpy import pi
from paddle import to_tensor, multiply
from paddle_quantum.mbqc.qobject import Pattern, Circuit
from paddle_quantum.mbqc.utils import div_str_to_float, int_to_div_str, print_progress

__all__ = [
    "MCalculus"
]


class MCalculus:
    r"""Define the class MCalculus.

    According to the paper's [The measurement calculus, arXiv: 0704.1263] measurement language, 
    this class provides series of basic operations dealing with  measurement patterns.
    """

    def __init__(self):
        r""" The constructors of ``MCalculus`` which is used for initialize a ``MCalculus`` object.
        """
        self.__circuit_slice = []  # Restore the information of sliced circuit
        self.__wild_pattern = []  # Record the background pattern
        self.__wild_commands = []  # Record wild commands information
        self.__pattern = None  # Record standard pattern
        self.__measured_qubits = []  # Record the measured qubits in the circuit model
        self.__circuit_width = None  # Record the circuit width
        self.__track = False  # Switch of progress bar

    def track_progress(self, track=True):
        r"""The button for revealing the progress bar during dealing with the measurement pattern.

        Args:
            track (bool, optional): ``True`` open the progress bar, ``False`` close the progress bar, the default is ``True``
        """
        assert isinstance(track, bool), "parameter 'track' must be a bool."
        self.__track = track

    def set_circuit(self, circuit):
        r"""Set the quantum circuit for the  ``MCalculus`` class.

        Args:
            circuit (Circuit): quantum circuit
        """
        assert isinstance(circuit, Circuit), "please input a parameter of type 'Circuit'."
        assert circuit.is_valid(), "the circuit is not valid as at least one qubit is not performed any gate yet."
        self.__circuit_width = circuit.get_width()
        self.__slice_circuit(circuit.get_circuit())
        for gate in self.__circuit_slice:
            self.__to_pattern(gate)
        self.__join_patterns()

    def __slice_circuit(self, circuit):
        r"""Cut the circuit into slices, label the input and output qubit of each quantum gate.

        Note:
            This is an internal method which the users don't need to use directly.
            Here we use``str`` type variables as the label of the nodes. In order to avoid losing the
            coordinates of the previous nodes and benefit for the plotting operation, we label
            all the nodes by the type looks like ``("1/1","2/1")``.

        Args:
            circuit (list): a list of circuits, each element represents a quantum gate or measurement.
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
        r"""Insert the auxiliary qubit

        Insert auxiliary qubit between the input and output qubit. The coordinates of the auxiliary qubits are
        assigned according to the qubit number, and the labels of them are the same as the input/output qubits.

        Note:

            This is an internal method which the users don't need to use directly.

        Args:
            input_ (list): The input node of the measurement pattern.
            output_ (list): The output node of the measurement pattern.
            ancilla_num_list (list): The list of the auxiliary qubits required to be inserted to the system.

        Returns:
            list: The list containing the labels of the auxiliary qubits.
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
        r""" Translate the quantum gates and quantum measurements to their equivalent measurement patterns in MBQC.

        Note:
            This is an internal method which the users don't need to use directly.

        Warning:
            The current version supports ``[H, X, Y, Z, S, T, Rx, Ry, Rz, Rz_5, U, CNOT, CNOT_15, CZ]`` quantum gates
            and single qubit measurement.
            By the way, there are no one-to-one correspondence between the quantum circuits and measurement patterns.
            Here, we adopt the most common onr or two measurement patterns.

        Args:
            gate (list): The quantum gates or quantum measurements to be translated. The elements in the list
                contains the original quantum gates(the name of gate, acting qubits, parameters),input/output qubits.
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
        r""" Change a 1-dimensional list containing the labels of qubits to a 2-dimensional list corresponding
        to the circuit diagram.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            bit_lst (list): the list containing the labels of the input/output nodes

        Returns:
            list: a 2-dimensional list with circuit structure, each line contains the column coordinates
            of the input/output nodes.
        """
        bit_net = [[] for _ in range(self.__circuit_width)]
        for qubit in bit_lst:
            bit_net[int(div_str_to_float(qubit[0]))].append(div_str_to_float(qubit[1]))
        return bit_net

    def __get_input(self, bit_lst):
        r"""Get the labels of the input qubits in the list.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            bit_lst (list): the list containing the labels of the nodes

        Returns:
            list: the list containing node labels of the input qubits
        """
        bit_net = self.__list_to_net(bit_lst)
        input_ = []
        for i in range(self.__circuit_width):
            input_.append((int_to_div_str(i), int_to_div_str(int(min(bit_net[i])))))
        return input_

    def __get_output(self, bit_lst):
        r""" Get the labels of the output qubits in the list.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            bit_lst (list): the list containing the labels of the nodes

        Returns:
            list: the list containing node labels of the output qubits
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
        r"""Connect the translated measurement patterns of the quantum gates and quantum measurements.

        Note:
            This is an internal method which the users don't need to use directly.
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
        r"""Swap two commands

        The swap of any two commands obeys the commutation relation of the operators, see
        more details in [arXiv:0704.1263].

        Note:
            This is an internal method which the users don't need to use directly.

        Hint:
            We default to the left-to-right order of operators, such as "E" before all
            operations, so "E" should be at the leftmost in the list of commands, and so on.

        Warning:
            This method exchange any two of the five commands `"E"``, ``"M"``, ``"X"``, ``"Z"``, ``"S"``.

        Args:
            which_cmds (list): 待交换的命令列表 The list of commands to be exchanged.

        Returns:
            list: The list of the commands after exchanged.
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
        r""" Move forward the commands of a specific type.

        The "move forward" should obeys the commutation rules. For example,
        the measurement operations should behind the entanglement operations
        acting on the same qubits as the measurements. When we use his method to move
        the measurement commands, the commands should be move forward until the first
        entanglement operation in front of them.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            cmd_type (str): the type of the command to be exchanged, 为 ``"E"``, ``"M"``, ``"X"``, ``"Z"`` 或 ``"S"``
            cmds (list): the list of commands to be dealt with.

        Returns:
            list: 交换后的新命令列表the new command list after exchange.
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
        r""" Sort the labels by the number of rows form small to large.

        The purpose of this method is to adjust the order of input/output nodes,
        so as to be consistent with the order of the circuit diagram and facilitate
        subsequent operations.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            labels (list): the list of labels to be dealt with.

        Returns:
            list: the list of labels sorted by the number of rows form small to large.
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
        r""" Map the list of commands to the list of numbers.

        The rule is CommandE -> 1, CommandM -> 2, CommandX -> 3, CommandZ -> 4, CommandS -> 5.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            cmds (list): The list of commands to be dealt with.

        Returns:
            list: contains the number of each type of command
            list: the list of numbers after the map
            list: the sorted standard number list after the map
        """
        cmd_map = {"E": 1, "M": 2, "X": 3, "Z": 4, "S": 5}
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_standard = cmd_num_wild[:]
        cmd_num_standard.sort(reverse=False)
        cmds_count = [cmd_num_standard.count(i) for i in [1, 2, 3, 4, 5]]  # Count each type of commands

        return cmds_count, cmd_num_wild, cmd_num_standard

    def __distance_to_standard(self, cmds):
        r""" Use Hamming distance to measure the difference between the order of the current list
            and the standard list.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            cmds (list): the current command list
        """
        _, cmd_wild, cmd_std = self.__commands_to_numbers(cmds[:])

        return sum([cmd_wild[i] == cmd_std[i] for i in range(len(cmd_wild))]) / len(cmd_wild)

    def __is_standard(self, cmd_type, cmds):
        r""" This method judges whether the given type of command is in the standard order.

        Note:
            This is an internal method which the users don't need to use directly.

        Args:
            cmd_type (str): The command to be judged, including ``E``, ``M``, ``X``, ``Z`` 或 ``S``
            cmds (list): the list of commands to be judged

        Returns:
            bool: the bool value of whether the list is in the standard order.
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
        r""" This method simplifies the dependence relation between different nodes.

        In some special cases, the dependence of the measurement nodes to other nodes can be simplified.
        Assume that \alpha is a measurement angle without considering the dependence relation, there are four
        possible measurement angles as:

         .. math::

            \theta_{\text{ad}} = \alpha

            \theta_{\text{ad}} = \alpha + \pi

            \theta_{\text{ad}} = - \alpha

            \theta_{\text{ad}} = - \alpha + \pi

            \text{when } \alpha \text{ is } 0, \pi / 2, \pi, 3 \times \pi / 2 \text{ the dependence relation can be simplified.}

            \text{For example, when } \alpha = \pi \text{, }\pm \alpha + t \times \pi \text{ lead to the same measurement result, 
            independent of domain\_s. As a result, domain\_s can be removed.}

        Note:
        In some special cases, the dependence of the measurement nodes to other nodes can be simplified.
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
        r"""Standardize the measurement pattern.

        This method standardize the measurement pattern to the equivalent EMC model. The method exchange all
        the "Command E" to the first, followed by "Command M", "Command X" and "Command Z". To simplify the
        measurement pattern, this method applies Pauli simplification to "Command M" after the standardization.
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
        r""" Extract the signal shifting operators from the measurement operators in the command list.

        Note:
            Signal shifting is a special operation. By switching with other command operators, this method
            remove the dependency among nodes in domain t in order to simplify the measurement patterns on some
            conditions, see more details in [arXiv:0704.1263].

        Warning:
            We only extract the node dependence of the nodes in domain t of measurement operators in XY plane, and
            we don't apply this operation to the measurement operator in YZ plane.

        Args:
            cmds (list): command list

        Returns:
            list: 提取信号转移算符后的命令列表 command list after extracting the signal shifting operators
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
        r"""Signal shifting operation

        Note:
            This is one of the choices of the users.
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
        r"""Return the measurement pattern

        Returns:
            Pattern: the measurement pattern after processing.
        """
        return self.__pattern

    @staticmethod
    def __default_order_by_row(labels):
        r""" Sort the nodes' labels by the number of rows.


        sorting rules: If the number of rows of the labels are different,
        smaller first; If the number of rows of the labels are the same, the labels
        with smaller number of columns first.


        Note:
            This is an internal method and users don't use it directly.

        Args:
            labels (list): the list of the nodes' labels to be processed.

        Returns:
            list: the list of nodes' labels after sorting.
        """
        # Construct a dict by string labels and their float values
        labels_dict = {label: (div_str_to_float(label[0]), div_str_to_float(label[1])) for label in labels}
        # Sort the dict by values (sort row first and then column)
        sorted_dict = dict(sorted(labels_dict.items(), key=lambda item: item[1]))
        # Extract the keys in the dict
        labels_sorted = list(sorted_dict.keys())

        return labels_sorted

    def optimize_by_row(self):
        r""" Optimize the measurement order in the measurement pattern according
        to row major order.

        Warning:
            This is a heuristic optimization algorithm works for specific measurement patterns.
            It's also possible to find better measurement order. For example, for a shallow quantum
            circuit, a specific qubit is no more needed after the quantum measurements according to the
            row major order such that the computational cost in the follow-up process is simplified.
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
