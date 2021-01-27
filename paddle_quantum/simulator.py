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
This simulator uses statevector(Tensor) to simulate quantum behaviors.
Basically, the core of the algorithm is tensor contraction with one-way calculation that each gate is
contracted to the init vector when imported by the program. All the states including the gate and init
are converted to TENSOR and the calculating is also around tensor.

:DEBUG INFO:
Sim2
-Sim2Main.py: this file, main entry of sim2
-InitProcess.py: Initial the state.
-StateTransfer.py: Decide the gate matrix by gate name and real state process.
-TransferProcess.py: Real transfer state process
-MeasureProcess.py: Measure process
Ancilla
-Random Circuit: Generate random circuit by requiring qubits and circuit depth
-DEFINE_GATE: Gate matrix.
Two measure types are provided: Meas_MED = MEAS_METHOD.PROB and Meas_MED = MEAS_METHOD.SINGLE. PROB is the sample with
probability and SINGLE is by the state collapse method. Former is significant faster than the later.
"""

import numpy as np
import paddle
import paddle.fluid
import gc
from collections import Counter
import copy
from interval import Interval
from enum import Enum
import random


### InitPorcess ###
def init_state_10(n):
    """
    Generate state with n qubits
    :param n: number of qubits
    :return: tensor of state
    """
    re1 = paddle.fluid.layers.ones([1], 'float64')
    re0 = paddle.fluid.layers.zeros([2 ** n - 1], 'float64')
    re = paddle.fluid.layers.concat([re1, re0])
    del re1, re0
    gc.collect()  # free the intermediate big data immediately
    im = paddle.fluid.layers.zeros([2 ** n], 'float64')
    state = paddle.fluid.ComplexVariable(re, im)
    del re, im
    gc.collect()  # free the intermediate big data immediately
    # print(state.numpy())

    return state


def init_state_gen(n, i = 0):
    """
    Generate state with n qubits
    :param n: number of qubits
    :param i: the ith vector in computational basis
    :return: tensor of state
    """
    assert 0 <= i < 2**n, 'Invalid index'

    if n == 1:
        re1 = paddle.fluid.layers.ones([1], 'float64')
        re0 = paddle.fluid.layers.zeros([2 ** n - 1], 'float64')

        if i == 0:
            re = paddle.fluid.layers.concat([re1, re0])
        else:
            re = paddle.fluid.layers.concat([re0, re1])
        im = paddle.fluid.layers.zeros([2 ** n], 'float64')
        state = paddle.fluid.ComplexVariable(re, im)
    else:
        if i == 0:
            re1 = paddle.fluid.layers.ones([1], 'float64')
            re0 = paddle.fluid.layers.zeros([2 ** n - 1], 'float64')
            re = paddle.fluid.layers.concat([re1, re0])
        elif i == 2 ** n - 1:
            re1 = paddle.fluid.layers.ones([1], 'float64')
            re0 = paddle.fluid.layers.zeros([2 ** n - 1], 'float64')
            re = paddle.fluid.layers.concat([re0, re1])
        else:
            re1 = paddle.fluid.layers.ones([1], 'float64')
            re0 = paddle.fluid.layers.zeros([i], 'float64')
            re00 = paddle.fluid.layers.zeros([2 ** n - i - 1], 'float64')
            re = paddle.fluid.layers.concat([re0, re1, re00])

        del re1, re0
        gc.collect()  # free the intermediate big data immediately
        im = paddle.fluid.layers.zeros([2 ** n], 'float64')
        state = paddle.fluid.ComplexVariable(re, im)
        del re, im
        gc.collect()  # free the intermediate big data immediately
    return state



### DEFINE_GATE ###
def x_gate_matrix():
    """
    Pauli x
    :return:
    """
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)


def y_gate_matrix():
    """
    Pauli y
    :return:
    """
    return np.array([[0, -1j],
                     [1j, 0]], dtype=complex)


def z_gate_matrix():
    """
    Pauli y
    :return:
    """
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)


def h_gate_matrix():
    """
    Hgate
    :return:
    """
    isqrt_2 = 1.0 / np.sqrt(2.0)
    return np.array([[isqrt_2, isqrt_2],
                     [isqrt_2, -isqrt_2]], dtype=complex)


def u_gate_matrix(params):
    """
    U3
    :param params:
    :return:
    """
    theta, phi, lam = params

    if (type(theta) is paddle.fluid.core_avx.VarBase and
            type(phi) is paddle.fluid.core_avx.VarBase and
            type(lam) is paddle.fluid.core_avx.VarBase):
        re_a = paddle.fluid.layers.cos(theta / 2)
        re_b = - paddle.fluid.layers.cos(lam) * paddle.fluid.layers.sin(theta / 2)
        re_c = paddle.fluid.layers.cos(phi) * paddle.fluid.layers.sin(theta / 2)
        re_d = paddle.fluid.layers.cos(phi + lam) * paddle.fluid.layers.cos(theta / 2)
        im_a = paddle.fluid.layers.zeros([1], 'float64')
        im_b = - paddle.fluid.layers.sin(lam) * paddle.fluid.layers.sin(theta / 2)
        im_c = paddle.fluid.layers.sin(phi) * paddle.fluid.layers.sin(theta / 2)
        im_d = paddle.fluid.layers.sin(phi + lam) * paddle.fluid.layers.cos(theta / 2)
        re = paddle.fluid.layers.reshape(paddle.fluid.layers.concat([re_a, re_b, re_c, re_d]), [2, 2])
        im = paddle.fluid.layers.reshape(paddle.fluid.layers.concat([im_a, im_b, im_c, im_d]), [2, 2])
        return paddle.fluid.framework.ComplexVariable(re, im)
    elif (type(theta) is float and
          type(phi) is float and
          type(lam) is float):
        return np.array([[np.cos(theta / 2),
                          -np.exp(1j * lam) * np.sin(theta / 2)],
                         [np.exp(1j * phi) * np.sin(theta / 2),
                          np.exp(1j * phi + 1j * lam) * np.cos(theta / 2)]])
    else:
        assert False


# compare the paddle and np version, they should be equal
# a = u_gate_matrix([1.0, 2.0, 3.0])
# print(a)
# with paddle.fluid.dygraph.guard():
#     a = u_gate_matrix([paddle.fluid.dygraph.to_variable(np.array([1.0])),
#                        paddle.fluid.dygraph.to_variable(np.array([2.0])),
#                        paddle.fluid.dygraph.to_variable(np.array([3.0]))])
#     print(a.numpy())


def cx_gate_matrix():
    """
    Control Not
    :return:
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex).reshape(2, 2, 2, 2)

def swap_gate_matrix():
    """
    Control Not
    :return:
    """
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex).reshape(2, 2, 2, 2)


### PaddleE ###
def normalize_axis(axis, ndim):
    if axis < 0:
        axis += ndim

    if axis >= ndim or axis < 0:
        raise ValueError("Invalid axis index %d for ndim=%d" % (axis, ndim))

    return axis


def _operator_index(a):
    return a.__index__()


def _normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [_operator_index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple([normalize_axis(ax, ndim) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError('repeated axis in `{}` argument'.format(argname))
        else:
            raise ValueError('repeated axis')
    return axis


def moveaxis(m, source, destination):
    """
    extend paddle
    :param m:
    :param source:
    :param destination:
    :return:
    """
    source = _normalize_axis_tuple(source, len(m.shape), 'source')
    destination = _normalize_axis_tuple(destination, len(m.shape), 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(len(m.shape)) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = paddle.fluid.layers.transpose(m, order)
    return result


def complex_moveaxis(m, source, destination):
    """
    extend paddle
    :param m:
    :param source:
    :param destination:
    :return:
    """
    source = _normalize_axis_tuple(source, len(m.shape), 'source')
    destination = _normalize_axis_tuple(destination, len(m.shape), 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(len(m.shape)) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = paddle.complex.transpose(m, order)
    return result


def complex_abs(m):
    # check1 = np.abs(m.numpy())

    re = paddle.fluid.layers.elementwise_mul(m.real, m.real)
    im = paddle.fluid.layers.elementwise_mul(m.imag, m.imag)
    m = paddle.fluid.layers.elementwise_add(re, im)
    m = paddle.fluid.layers.sqrt(m)  # m = paddle.fluid.layers.elementwise_pow(m, paddle.fluid.layers.ones_like(m) * 0.5)

    # check2 = m.numpy()
    # assert (check1 == check2).all()

    return m


### TransferProcess ###
def transfer_state(state, gate_matrix, bits):
    """
    Transfer to the next state
    :param state:
    :param gate_matrix:
    :param bits:
    :return:
    """

    assert type(gate_matrix) is np.ndarray or type(gate_matrix) is paddle.fluid.framework.ComplexVariable

    assert type(state) is paddle.fluid.ComplexVariable and len(state.shape) == 1
    # calc source_pos target_pos
    n = int(np.log2(state.shape[0]))
    source_pos = copy.deepcopy(bits)  # copy bits, it should NOT change the order of bits
    # source_pos = [n - 1 - idex for idex in source_pos]  # qubit index
    # source_pos = list(reversed(source_pos))  # reverse qubit index
    target_pos = list(range(len(bits)))

    # ### check
    # state_check = transfer_state(paddle.complex.reshape(state, [2] * n), gate_matrix, bits)
    # state_check = paddle.complex.reshape(state_check, [2 ** n])

    # compressed moveaxis
    # compress the continuous dim before moveaxis
    # e.g. single operand: before moveaxis 2*2*[2]*2*2 -compress-> 4*[2]*4, after moveaxis [2]*2*2*2*2 -compress-> [2]*4*4
    #      double operands: before moveaxis 2*2*[2]*2*2*[2]*2*2 -compress-> 4*[2]*4*[2]*4, after moveaxis [2]*[2]*2*2*2*2*2*2 -compress-> [2]*[2]*4*4*4
    # the peak rank is 5 when the number of operands is 2
    assert len(source_pos) == 1 or len(source_pos) == 2
    compressed_shape_before_moveaxis = [1]
    compressed_source_pos = [-1] * len(source_pos)
    for i in range(n):
        if i in source_pos:
            compressed_source_pos[source_pos.index(i)] = len(compressed_shape_before_moveaxis)
            compressed_shape_before_moveaxis.append(2)
            compressed_shape_before_moveaxis.append(1)
        else:
            compressed_shape_before_moveaxis[-1] = compressed_shape_before_moveaxis[-1] * 2
    # print([2] * n)
    # print(source_pos)
    # print('->')
    # print(compressed_shape)
    # print(compressed_source_pos)  # always [1], [1, 3], or [3, 1]
    state = paddle.complex.reshape(state, compressed_shape_before_moveaxis)
    state = complex_moveaxis(state, compressed_source_pos, target_pos)
    compressed_shape_after_moveaxis = state.shape

    # reshape
    state_new_shape = [2 ** len(bits), 2 ** (n - len(bits))]
    state = paddle.complex.reshape(state, state_new_shape)

    # gate_matrix
    if type(gate_matrix) is np.ndarray:
        gate_new_shape = [2 ** (len(gate_matrix.shape) - len(bits)), 2 ** len(bits)]
        gate_matrix = gate_matrix.reshape(gate_new_shape)
        gate_matrix = paddle.fluid.dygraph.to_variable(gate_matrix)
    elif type(gate_matrix) is paddle.fluid.framework.ComplexVariable:
        pass
    else:
        assert False

    # matmul
    state = paddle.complex.matmul(gate_matrix, state)

    # restore compressed moveaxis reshape
    state = paddle.complex.reshape(state, compressed_shape_after_moveaxis)
    state = complex_moveaxis(state, target_pos, compressed_source_pos)
    state = paddle.complex.reshape(state, [2 ** n])

    # ### check
    # assert (np.all(state.numpy() == state_check.numpy()))

    return state


### StateTranfer ###
def StateTranfer(state, gate_name, bits, params=None):
    """
    To transfer state by only gate name and bits
    :param state: the last step state, can be init vector or  the last step vector.
    :param gate_name:x,y,z,h,CNOT, SWAP
    :param bits: the gate working on the bits.
    :param params: params for u gate.
    :return: the updated state
    """
    if gate_name == 'h':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = h_gate_matrix()
    elif gate_name == 'x':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = x_gate_matrix()
    elif gate_name == 'y':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = y_gate_matrix()
    elif gate_name == 'z':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = z_gate_matrix()
    elif gate_name == 'CNOT':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = cx_gate_matrix()
    elif gate_name == 'SWAP':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = swap_gate_matrix()
    elif gate_name == 'u':
        # print('----------', gate_name, bits, '----------')
        gate_matrix = u_gate_matrix(params)
    else:
        raise Exception("Gate name error")

    state = transfer_state(state, gate_matrix, bits)
    return state


### MeasureProcess ###
class MEAS_METHOD(Enum):
    """
    To control the measure method
    """
    SINGLE = 1
    PROB = 2


Meas_MED = MEAS_METHOD.PROB


def oct_to_bin_str(oct_number, n):
    """
    Oct to bin by real order
    :param oct_number:
    :param n:
    :return:
    """
    bin_string = bin(oct_number)[2:].zfill(n)
    # return (''.join(reversed(bin_string)))
    return bin_string


def measure_single(state, bit):
    """
    Method one qubit one time
    :param state:
    :param bit:
    :return:
    """
    n = len(state.shape)
    axis = list(range(n))
    axis.remove(n - 1 - bit)
    probs = np.sum(np.abs(state) ** 2, axis=tuple(axis))
    rnd = np.random.rand()

    # measure single bit
    if rnd < probs[0]:
        out = 0
        prob = probs[0]
    else:
        out = 1
        prob = probs[1]

    # collapse single bit
    if out == 0:
        matrix = np.array([[1.0 / np.sqrt(prob), 0.0],
                           [0.0, 0.0]], complex)
    else:
        matrix = np.array([[0.0, 0.0],
                           [0.0, 1.0 / np.sqrt(prob)]], complex)
    state = transfer_state(state, matrix, [bit])

    return out, state


def measure_all(state):
    """
    Method all by single qubits
    :param state:
    :return:
    """
    n = len(state.shape)
    outs = ''
    for i in range(n):
        out, state = measure_single(state, i)  # measure qubit0 will collapse it
        outs = str(out) + outs                 # from low to high position
    return outs


def measure_by_single_accumulation(state, shots):
    """
    Method by accumulation, one shot one time
    :param state:
    :param shots:
    :return:
    """
    print("Measure method Single Accu")
    result = {}
    for i in range(shots):
        outs = measure_all(state)
        if outs not in result:
            result[outs] = 0
        result[outs] += 1
    return result


def measure_by_probability(state, times):
    """
    Measure by probability method
    :param state:
    :param times:
    :return:
    """
    # print("Measure method Probability")

    assert type(state) is paddle.fluid.ComplexVariable and len(state.shape) == 1
    n = int(np.log2(state.shape[0]))
    prob_array = complex_abs(state)  # complex -> real
    prob_array = paddle.fluid.layers.elementwise_mul(prob_array, prob_array)
    prob_array = prob_array.numpy()
    gc.collect()

    """
    prob_key = []
    prob_values = []
    pos_list = list(np.nonzero(prob_array)[0])
    for index in pos_list:
        string = oct_to_bin_str(index, n)
        prob_key.append(string)
        prob_values.append(prob_array[index])

    # print("The sum prob is ", sum(prob_values))

    samples = np.random.choice(len(prob_key), times, p=prob_values)
    """
    samples = np.random.choice(range(2 ** n), times, p=prob_array)
    count_samples = Counter(samples)
    result = {}
    for idex in count_samples:
        """
        result[prob_key[idex]] = count_samples[idex]
        """
        result[oct_to_bin_str(idex, n)] = count_samples[idex]
    return result


def measure_state(state, shots):
    """
    Measure main entry
    :param state:
    :param shots:
    :return:
    """
    if Meas_MED == MEAS_METHOD.SINGLE:
        return measure_by_single_accumulation(state, shots)
    elif Meas_MED == MEAS_METHOD.PROB:
        return measure_by_probability(state, shots)
    else:
        raise Exception("Measure Error")


### RandomCircuit ###
def GenerateRandomCirc(state, circ_depth, n):
    """
    Generate random circ
    :param state: The whole state
    :param circ_depth: number of circuit
    :param n: number of qubits
    :return: state
    """
    gate_string = ['x', 'y', 'z', 'h', 'CNOT']
    internal_state = state
    for index in range(circ_depth):
        rand_gate_pos = random.randint(0, len(gate_string) - 1)
        if rand_gate_pos == (len(gate_string) - 1):
            rand_gate_bits = random.sample(range(n), 2)
        else:
            rand_gate_bits = random.sample(range(n), 1)
        internal_state = StateTranfer(internal_state, gate_string[rand_gate_pos], rand_gate_bits)
    state = internal_state
    return state


def GenerateRandomCircAndRev(state, circ_depth, n):
    """
    Generate random circ
    :param state: The whole state
    :param circ_depth: number of circuit
    :param n: number of qubits
    :return: state
    """
    rand_gate_pos_all = []
    rand_gate_bits_all = []
    gate_string = ['x', 'y', 'z', 'h', 'CNOT']
    internal_state = state
    for index in range(circ_depth):
        rand_gate_pos = random.randint(0, len(gate_string) - 1)
        if rand_gate_pos == (len(gate_string) - 1):
            rand_gate_bits = random.sample(range(n), 2)
        else:
            rand_gate_bits = random.sample(range(n), 1)

        rand_gate_pos_all.append(rand_gate_pos)
        rand_gate_bits_all.append(rand_gate_bits)

        internal_state = StateTranfer(internal_state, gate_string[rand_gate_pos], rand_gate_bits)

    rand_gate_pos_all = list(reversed(rand_gate_pos_all))
    rand_gate_bits_all = list(reversed(rand_gate_bits_all))

    for idex, item in enumerate(rand_gate_pos_all):
        internal_state = StateTranfer(internal_state, gate_string[item], rand_gate_bits_all[idex])

    state = internal_state
    return state


### Tester ###
def RandomTestIt(bits_num, circ_num):
    """
    Random Check
    :param bits_num:
    :param circ_num:
    :return:
    """
    n = bits_num
    repeat_num = 1  # 2 ** 5
    state = init_state_10(n)

    for _ in range(repeat_num):
        # state = copy.deepcopy(state_origin)
        state = GenerateRandomCircAndRev(state, circ_num, n)
        re = measure_state(state, 2 ** 10)
        # print(re)
        assert (str('0' * n) in list(re.keys()))
        assert (2 ** 10 in list(re.values()))

    return True


def Test_x_h_CNOT():  # 3 bits coverage test
    """
    The smallest tester program
    :return:
    """
    state = init_state_10(3)

    state = StateTranfer(state, 'x', [0])
    state = StateTranfer(state, 'h', [1])
    state = StateTranfer(state, 'CNOT', [1, 2])
    re = measure_state(state, 2 ** 10)
    # print(re)
    assert ('001' in list(re.keys()) and '111' in list(re.keys()))
    assert (list(re.values())[0] / list(re.values())[1] in Interval(0.9, 1.1))
    return True


def Test_cnot_1():
    """
    Check CNOT using reverse circ
    :return:
    """
    state = init_state_10(3)

    state = StateTranfer(state, 'h', [0])
    state = StateTranfer(state, 'CNOT', [0, 1])
    state = StateTranfer(state, 'CNOT', [0, 2])
    state = StateTranfer(state, 'CNOT', [1, 2])
    state = StateTranfer(state, 'CNOT', [1, 2])
    state = StateTranfer(state, 'CNOT', [0, 2])
    state = StateTranfer(state, 'CNOT', [0, 1])
    state = StateTranfer(state, 'h', [0])
    re = measure_state(state, 2 ** 10)
    assert ('000' in list(re.keys()))
    assert (2 ** 10 in list(re.values()))
    return True


def Test_cnot_2():
    """
    Check retation of CNOTS
    :return:
    """
    state = init_state_10(3)

    state = StateTranfer(state, 'h', [2])
    state = StateTranfer(state, 'CNOT', [2, 1])
    state = StateTranfer(state, 'CNOT', [2, 0])
    state = StateTranfer(state, 'CNOT', [1, 0])
    state = StateTranfer(state, 'CNOT', [1, 0])
    state = StateTranfer(state, 'CNOT', [2, 0])
    state = StateTranfer(state, 'CNOT', [2, 1])
    state = StateTranfer(state, 'h', [2])
    re = measure_state(state, 2 ** 10)
    assert ('000' in list(re.keys()))
    assert (2 ** 10 in list(re.values()))
    return True


def Test3All():
    """
    Check all 3 qubits cases
    :return:
    """
    return Test_x_h_CNOT() and Test_cnot_1() and Test_cnot_2()


### Sim2Main ###
def main(bits_num=None, circ_num=None):
    """
    :param bits_num:  the number of qubits
    :param circ_num: the circuit depth to be expected
    These two args can be None type and the default behavior is bits_num =5 and circ_num=50.
     Two args must be given together.
    :return:
    :DEBUG INFO:
    1) np.random.seed(0) can be uncomment to guarantee the random sequence can be tracked.
    2) re is the return result: vector string: counter
    """

    # init seed
    # np.random.seed(0)

    print('----------', 'init', bits_num, 'bits', '----------')
    print('----------', 'init', circ_num, 'circ depth', '----------')

    state = init_state_10(bits_num)

    # ------------------tik-------
    # state = StateTranfer(state, 'h', [0])
    # state = StateTranfer(state, 'x', [1])
    # state = StateTranfer(state, 'x', [2])
    # state = StateTranfer(state, 'u', [3], [0.3, 0.5, 0.7])
    # state = StateTranfer(state, 'CNOT', [0, 1])
    # state = StateTranfer(state, 'CNOT', [2, 1])
    # -----------------tok---------

    # ------------------tik-------
    state = StateTranfer(state, 'x', [0])
    state = StateTranfer(state, 'h', [1])
    #state = StateTranfer(state, 'CNOT', [0, 1])
    # -----------------tok---------

    # ------------------tik-------
    # state = GenerateRandomCirc(state, circ_num, n)
    # -----------------tok---------

    # ------------------tik-------
    re = measure_state(state, 2 ** 10)
    # -----------------tok---------

    print(re)


def Tester(bits_num=None, circ_num=None):
    """
    This part is to guarantee the behaviors of the simualtor2. Every Modification MUST ensure this function can be executed correctly.
    :return: True: Pass or False: NO Pass
    """
    # random tester

    check_value_rand = RandomTestIt(bits_num, circ_num)
    check_value_3 = Test3All()
    check_all = check_value_rand and check_value_3
    print(check_all)
    return check_all


if __name__ == '__main__':
    with paddle.fluid.dygraph.guard():
        # main(bits_num=5, circ_num=50)
        main(bits_num=5)
        # RandomTestIt(bits_num=5, circ_num=50)
        # print(Test3All())

        # print(Tester(bits_num=5, circ_num=50))
        #Tester(bits_num=6, circ_num=100)
