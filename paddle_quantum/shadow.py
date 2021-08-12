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
shadow sample module
"""

import numpy as np
import paddle
import re
from paddle_quantum import circuit
from paddle_quantum.utils import Hamiltonian

__all__ = [
    "shadow_sample"
]


def shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian=None, method='CS'):
    r"""对给定的量子态进行随机的泡利测量并返回测量结果。

    Args:
        state (numpy.ndarray): 输入量子态，支持态矢量和密度矩阵形式
        num_qubits (int): 量子比特数量
        sample_shots (int): 随机采样的次数
        mode (str): 输入量子态的表示方式，``"state_vector"`` 表示态矢量形式， ``"density_matrix"`` 表示密度矩阵形式
        hamiltonian (Hamiltonian, optional): 可观测量的相关信息，输入形式为 ``Hamiltonian`` 类，默认为 ``None``
        method (str, optional): 进行随机采样的方法，有 ``"CS"`` 、 ``"LBCS"`` 、 ``"APS"`` 三种方法，默认为 ``"CS"``

    Returns:
        list: 随机选择的泡利测量基和测量结果，形状为 ``(sample_shots, 2)`` 的list

    代码示例:

    .. code-block:: python
        from paddle_quantum.shadow import shadow_sample
        from paddle_quantum.state import vec_random
        from paddle_quantum.utils import Hamiltonian

        n_qubit = 2
        sample_shots = 10
        state = vec_random(n_qubit)
        sample_data_cs = shadow_sample(state, n_qubit, sample_shots, mode='state_vector')

        ham = [[0.1, 'x1'], [0.2, 'y0']]
        ham = Hamiltonian(ham)
        sample_data_lbcs, beta_lbcs = shadow_sample(state, n_qubit, sample_shots, 'state_vector', "LBCS", ham)
        sample_data_aps = shadow_sample(state, n_qubit, sample_shots, 'state_vector', "APS", ham)

        print('sample data CS = ', sample_data_cs)
        print('sample data LBCS = ', sample_data_lbcs)
        print('beta LBCS = ', beta_lbcs)
        print('sample data APS = ', sample_data_aps)

    ::

        sample data CS =  [('zy', '10'), ('yx', '01'), ('zx', '01'), ('xz', '00'), ('zy', '11'), ('xz', '00'), ('xz', '11'), ('yy', '01'), ('yx', '00'), ('xx', '00')]
        sample data LBCS =  [('yx', '00'), ('yx', '01'), ('yx', '00'), ('yx', '01'), ('yx', '00'), ('yx', '01'), ('yx', '01'), ('yx', '00'), ('yx', '01'), ('yx', '01')]
        beta LBCS =  [[2.539244934862217e-05, 0.9999492151013026, 2.539244934862217e-05], [0.9999492151013026, 2.539244934862217e-05, 2.539244934862217e-05]]
        sample data APS =  [('yx', '10'), ('yx', '01'), ('yx', '00'), ('yx', '01'), ('yx', '10'), ('yx', '01'), ('yx', '10'), ('yx', '10'), ('yx', '00'), ('yx', '01')]

    """

    if hamiltonian is not None:
        if isinstance(hamiltonian, Hamiltonian):
            hamiltonian = hamiltonian.pauli_str

    def prepare_hamiltonian(hamiltonian, num_qubits):
        r"""改写可观测量[[0.3147,'y2'], [-0.5484158742278,'x2,z1'],...]的形式

        Args:
            hamiltonian (list): 可观测量的相关信息
            num_qubits (int): 量子比特数目

        Returns:
            list: 可观测量的形式改写为[[0.3147,'iiy'], [-0.5484158742278,'izx'],...]

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        new_hamiltonian = list()
        for idx, (coeff, pauli_term) in enumerate(hamiltonian):
            pauli_term = re.split(r',\s*', pauli_term.lower())
            pauli_list = ['i'] * num_qubits
            for item in pauli_term:
                if len(item) > 1:
                    pauli_list[int(item[1:])] = item[0]
                elif item[0].lower() != 'i':
                    raise ValueError('Expecting I for ', item[0])
            new_term = [coeff, ''.join(pauli_list)]
            new_hamiltonian.append(new_term)
        return new_hamiltonian

    if hamiltonian is not None:
        hamiltonian = prepare_hamiltonian(hamiltonian, num_qubits)

    pauli2index = {'x': 0, 'y': 1, 'z': 2}

    def random_pauli_sample(num_qubits, beta=None):
        r"""根据概率分布 beta, 随机选取 pauli 测量基

        Args:
            num_qubits (int): 量子比特数目
            beta (list, optional): 量子位上不同 pauli 测量基的概率分布

        Returns:
            str: 返回随机选择的 pauli 测量基

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        # assume beta obeys a uniform distribution if it is not given
        if beta is None:
            beta = list()
            for _ in range(0, num_qubits):
                beta.append([1 / 3] * 3)
        pauli_sample = str()
        for qubit_idx in range(num_qubits):
            sample = np.random.choice(['x', 'y', 'z'], 1, p=beta[qubit_idx])
            pauli_sample += sample[0]
        return pauli_sample

    def measure_by_pauli_str(pauli_str, phi, num_qubits, method):
        r"""搭建 pauli 测量电路，返回测量结果

        Args:
            pauli_str (str): 输入的是随机选取的num_qubits pauli 测量基
            phi (numpy.ndarray): 输入量子态，支持态矢量和密度矩阵形式
            num_qubits (int): 量子比特数量
            method (str): 进行测量的方法，有 "CS"、"LBCS"、"APS"

        Returns:
            str: 返回测量结果

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        if method == "clifford":
            # Add the clifford function
            pass
        else:
            # Other method are transformed as follows
            # Convert to tensor form
            input_state = paddle.to_tensor(phi)
            cir = circuit.UAnsatz(num_qubits)
            for qubit in range(num_qubits):
                if pauli_str[qubit] == 'x':
                    cir.h(qubit)
                elif pauli_str[qubit] == 'y':
                    cir.h(qubit)
                    cir.s(qubit)
                    cir.h(qubit)
            if mode == 'state_vector':
                cir.run_state_vector(input_state)
            else:
                cir.run_density_matrix(input_state)
            result = cir.measure(shots=1)
            bit_string, = result
            return bit_string

    # Define the function used to update the beta of the LBCS algorithm
    def calculate_diagonal_product(pauli_str, beta):
        r"""迭代 LBCS beta 公式中的一部分

        Hint:
            计算 \prod_{j \in supp(Q)} \beta_{j}(Q_{j})^{-1}

        Args:
            pauli_str (str): 输入的是 hamiltonian 的 pauli 项
            beta (list): 量子位上不同 pauli 测量基的概率分布

        Returns:
            float: 返回计算值

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        product = 1
        for qubit_idx in range(len(pauli_str)):
            if pauli_str[qubit_idx] != 'i':
                index = pauli2index[pauli_str[qubit_idx]]
                b = beta[qubit_idx][index]
                if b == 0:
                    return float('inf')
                else:
                    product *= b

        return 1 / product

    def lagrange_restriction_numerator(qubit_idx, hamiltonian, beta):
        r"""迭代 LBCS beta 公式中的分子

        Hint:
            计算 \sum_{Q \mid Q_{i}=P_{i}} \alpha_{Q}^{2} \prod_{j \in supp(Q)} \beta_{j}(Q_{j})^{-1}

        Args:
            qubit_idx (int): 第 qubit_idx 个量子位
            hamiltonian (list): 可观测量的相关信息
            beta (list): 量子位上不同 pauli 测量基的概率分布

        Returns:
            list: 返回第 qubit_idx 个量子位上不同 pauli 算子的在该式下的数值

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        tally = [0, 0, 0]
        for coeff, pauli_term in hamiltonian:
            if pauli_term[qubit_idx] == 'x':
                tally[0] += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
            elif pauli_term[qubit_idx] == 'y':
                tally[1] += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
            elif pauli_term[qubit_idx] == 'z':
                tally[2] += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
        return tally

    def lagrange_restriction_denominator(qubit_idx, random_observable, beta):
        r"""迭代 LBCS beta 公式中的分母

        Hint:
            计算 \sum_{Q \mid Q_{i} \neq I} \alpha_{Q}^{2} \prod_{j \in supp(Q)} \beta_{j}(Q_{j})^{-1}

        Args:
            qubit_idx (int): 第 qubit_idx 个量子位
            random_observable (list): 可观测量的相关信息
            beta (list): 量子位上不同 pauli 测量基的概率分布

        Returns:
            float: 返回第 qubit_idx 个量子位上在该式下的数值

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        tally = 0.0
        for coeff, pauli_term in random_observable:
            if pauli_term[qubit_idx] != "i":
                tally += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
        if tally == 0.0:
            tally = 1
        return tally

    def lagrange_restriction(qubit_idx, hamiltonian, beta, denominator=None):
        r"""迭代 LBCS beta 公式，将分子与分母结合起来

        Args:
            qubit_idx (int): 第 qubit_idx 个量子位
            hamiltonian (list): 可观测量的相关信息
            beta (list): 量子位上不同 pauli 测量基的概率分布
            denominator (float, optional): 迭代公式的分母，可默认为None

        Returns:
            list: 返回第 qubit_idx 个量子位上在该式下的数值

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        numerator = lagrange_restriction_numerator(qubit_idx, hamiltonian, beta)
        if denominator is None:
            denominator = lagrange_restriction_denominator(qubit_idx, hamiltonian, beta)
        return [item / denominator for item in numerator]

    def beta_distance(beta1, beta2):
        r"""计算迭代前后 beta 差距，以便停止迭代

        Args:
            beta1 (list): 迭代后的全部量子位上的概率分布
            beta2 (list): 迭代前的全部量子位上的概率分布

        Returns:
            numpy.float: 返回迭代前后 beta 差距

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        two_norm_squared = 0.0
        for qubit in range(len(beta1)):
            two_norm_squared_qubit = np.sum((np.array(beta1[qubit]) - np.array(beta2[qubit])) ** 2)
            two_norm_squared += two_norm_squared_qubit
        return np.sqrt(two_norm_squared)

    def update_beta_in_lbcs(hamiltonian, num_qubit, beta_old=None, weight=0.1):
        r"""LBCS 的 beta 迭代函数

        Args:
            hamiltonian (list): 可观测量的相关信息
            num_qubit (int): 量子比特数
            beta_old (list): 迭代前的全部量子位上的概率分布
            weight (float): 更新的步长，可根据需要修改

        Returns:
            list: 返回更新后的 beta
            numpy.float: 返回迭代前后 beta 差距

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        if beta_old is None:
            beta_old = list()
            for _ in range(0, num_qubit):
                beta_old.append([1 / 3] * 3)

        beta_new = list()
        for qubit in range(num_qubit):
            denominator = lagrange_restriction_denominator(qubit, hamiltonian, beta_old)
            lagrange_rest = lagrange_restriction(qubit, hamiltonian, beta_old, denominator)
            beta_new.append(lagrange_rest)
            if sum(beta_new[qubit]) != 0:
                beta_new[qubit] = [item / sum(beta_new[qubit]) for item in beta_new[qubit]]
            else:
                beta_new[qubit] = beta_old[qubit]
            for idx in range(len(beta_new[qubit])):
                beta_new[qubit][idx] = (1 - weight) * beta_old[qubit][idx] + weight * beta_new[qubit][idx]
        return beta_new, beta_distance(beta_new, beta_old)

    # Define the function used to update the beta of the APS algorithm
    def in_omega(pauli_str, qubit_idx, qubit_shift, base_shift):
        r"""用于判断 hamiltonian 的 pauli 项是否属于集合Omega

        Args:
            pauli_str (str): 可观测量的 pauli 项
            qubit_idx (int): 第 qubit_idx 量子位
            qubit_shift (list): 乱序重排量子位，比如第1位映射到第4位
            base_shift (float): 乱序排放的 pauli 测量基

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        if pauli_str[qubit_shift[qubit_idx]] == 'i':
            return False
        for former_qubit in range(qubit_idx):
            idx = qubit_shift[former_qubit]
            if not pauli_str[idx] in ('i', base_shift[former_qubit]):
                return False
        return True

    def update_in_aps(qubit_idx, qubits_shift, bases_shift, hamiltonian):
        r"""用于更新 APS 的 beta

        Args:
            qubit_idx (int): 第 qubit_idx 量子位
            qubit_shift (list): 乱序重排量子位，比如第1位映射到第4位
            base_shift (float): 乱序排放的 pauli 测量基
            hamiltonian (list): 可观测量的相关信息

        Returns:
            list: 返回更新后的 beta

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        constants = [0.0, 0.0, 0.0]
        for coeff, pauli_term in hamiltonian:
            if in_omega(pauli_term, qubit_idx, qubits_shift, bases_shift):
                pauli = pauli_term[qubits_shift[qubit_idx]]
                index = pauli2index[pauli]
                constants[index] += coeff ** 2
        beta_sqrt = np.sqrt(constants)
        # The beta may be zero, use a judgment statement to avoid
        if np.sum(beta_sqrt) == 0.0:
            beta = [1 / 3, 1 / 3, 1 / 3]
        else:
            beta = beta_sqrt / np.sum(beta_sqrt)
        return beta

    def single_random_pauli_sample_in_aps(qubit_idx, qubits_shift, pauli_str_shift, hamiltonian):
        r"""用于在单量子位上根据概率分布随机选取 pauli 测量基

        Args:
            qubit_idx (int): 第 qubit_idx 量子位
            qubit_shift (list): 乱序重排量子位，比如第1位映射到第4位
            base_shift (float): 乱序排放的 pauli 测量基
            hamiltonian (list): 可观测量的相关信息

        Returns:
            str: 返回在第 qubit_idx 量子位上选取的 pauli 测量基

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert len(pauli_str_shift) == qubit_idx
        beta = update_in_aps(qubit_idx, qubits_shift, pauli_str_shift, hamiltonian)
        single_pauli = np.random.choice(['x', 'y', 'z'], p=beta)
        return single_pauli

    def random_pauli_sample_in_aps(hamiltonian):
        r"""用于根据概率分布随机选择所有量子位上的 pauli 测量基

        Args:
            hamiltonian (list): 可观测量的相关信息

        Returns:
            list: 返回所有量子位上随机选择的 pauli 测量基

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        num_qubits = len(hamiltonian[0][1])
        # The qubits_shift is used to ignore the order of the qubits
        qubits_shift = list(np.random.choice(range(num_qubits), size=num_qubits, replace=False))
        pauli_str_shift = list()
        for qubit_idx in range(num_qubits):
            single_pauli = single_random_pauli_sample_in_aps(qubit_idx, qubits_shift, pauli_str_shift, hamiltonian)
            pauli_str_shift.append(single_pauli)
        pauli_sample = str()
        for i in range(num_qubits):
            j = qubits_shift.index(i)
            # The qubits_shift.index(i) sorts the qubits in order
            pauli_sample = pauli_sample + pauli_str_shift[j]
        return pauli_sample

    sample_result = list()
    if method == "CS":
        for _ in range(sample_shots):
            random_pauli_str = random_pauli_sample(num_qubits, beta=None)
            measurement_result = measure_by_pauli_str(random_pauli_str, state, num_qubits, method)
            sample_result.append((random_pauli_str, measurement_result))
        return sample_result
    elif method == "LBCS":
        beta = list()
        for _ in range(0, num_qubits):
            beta.append([1 / 3] * 3)
        beta_opt_iter_num = 10000
        distance_limit = 1.0e-6
        for _ in range(beta_opt_iter_num):
            beta_opt, distance = update_beta_in_lbcs(hamiltonian, num_qubits, beta)
            beta = beta_opt
            if distance < distance_limit:
                break
        sample_result = list()
        for _ in range(sample_shots):
            random_pauli_str = random_pauli_sample(num_qubits, beta)
            measurement_result = measure_by_pauli_str(random_pauli_str, state, num_qubits, method)
            sample_result.append((random_pauli_str, measurement_result))
        return sample_result, beta
    elif method == "APS":
        for _ in range(sample_shots):
            random_pauli_str = random_pauli_sample_in_aps(hamiltonian)
            measurement_result = measure_by_pauli_str(random_pauli_str, state, num_qubits, method)
            sample_result.append((random_pauli_str, measurement_result))
        return sample_result
