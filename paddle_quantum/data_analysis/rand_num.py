# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

import os
import logging
import paddle_quantum as pq
from QCompute import BackendName
import paddle
from math import log

r"""
Quantum random number generator.
"""

def random_number_generation(
        bit_len: int,
        backend: str = 'local_baidu_sim2',
        token: str = None,
        extract: bool = False,
        security: float = 1e-8,
        min_entr_1: float = 0.9,
        min_entr_2: float = 0.9,
        log_path: str = None):
    r"""
    An encapsuled method of the random number generation, 
        referring to the paper https://arxiv.org/abs/1311.5322.

    Args:
        bit_len: the count of numbers you needed
        backend: the physical processor, including 
                    'local_baidu_sim2',
                    'cloud_baidu_sim2_water',
                    'cloud_baidu_sim2_earth',
                    'cloud_baidu_sim2_thunder',
                    'cloud_baidu_sim2_heaven',
                    'cloud_baidu_sim2_wind',
                    'cloud_baidu_sim2_lake',
                    'cloud_aer_at_bd',
                    'cloud_baidu_qpu_qian',
                    'cloud_iopcas',
                    'cloud_ionapm',
                    'service_ubqc'.
        token: user's token for cloud service
        extract: whether to use extractor for post-process
        security: security parameters
        min_entr_1: the min-entropy of hardware 1
        min_entr_2: the min-entropy of hardware 2
        log_path: the save path of log file
    """

    # logger configuration
    if not log_path:
        log_path = './qrng.log'
    logger = logging.Logger(name='logger_random_number_generation')
    logger_file_handler = logging.FileHandler(log_path)
    logger_file_handler.setFormatter(logging.Formatter(r'%(levelname)s  %(asctime)s  %(message)s'))
    logger_file_handler.setLevel(logging.INFO)
    logger.addHandler(logger_file_handler)

    # environment configuration
    logger.info('QRNG Initializing...')
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    pq.set_backend('quleaf')
    pq.backend.quleaf.set_quleaf_backend(backend)
    if backend not in (BackendName.LocalBaiduSim2, 'local_baidu_sim2'):
        assert token, 'Cloud service needs token!'
    pq.backend.quleaf.set_quleaf_token(token)

    # calculation
    logger.info('Parameters:')
    logger.info(f'      Target bit length: {bit_len}')
    logger.info(f'      Backend: {backend}')
    logger.info(f'      Extractor state: {extract}')
    bit_string = []
    if not extract:
        cir = pq.ansatz.Circuit(num_qubits=1)
        cir.h()
        output_state = cir()
        for _ in range(bit_len):
            result = list(list(output_state.measure(shots=1,))[0])
            bit_string.append(int(result[0]))

        logger.info('Processing...')
        logger.info(f'Output: {bit_string}')
        return bit_string

    else:
        logger.info(f'      Security parameter: {security}')
        logger.info(f'      Min-entropy-1: {min_entr_1}')
        logger.info(f'      Min-entropy-2: {min_entr_2}')
        logger.info('Computing raw bits needed')
        shots = int((bit_len-1-2*log(security))/(min_entr_1 + min_entr_2 - 1))   # raw bit length needed
        logger.info(f'      Raw bit length: {shots}')
        print('need raw bit length: {shots}')

        cir = pq.ansatz.Circuit(num_qubits=2)
        cir.h()
        output_state = cir()

        bit_string1 = []
        bit_string2 = []
        for _ in range(shots):
            bit1, bit2 = list(list(output_state.measure(shots=1))[0])
            bit_string1.append(int(bit1[0]))
            bit_string2.append(int(bit2[0]))

        logger.info(f'      Bit string on hardware-1: {bit_string1}')
        logger.info(f'      Bit string on hardware-2: {bit_string2[:shots-1]}')
        logger.info('Processing...')

        string_vec1 = paddle.to_tensor(bit_string1, dtype='float32')
        string_vec2 = paddle.to_tensor(bit_string2[:shots-1], dtype='float32')

        topelitz = []
        for i in range(bit_len):
            topelitz.append(string_vec2[bit_len-1-i:shots-1-i])
        topelitz = paddle.concat((paddle.to_tensor(topelitz), paddle.ones((bit_len, bit_len))), axis=1)
        extract = paddle.matmul(string_vec1.unsqueeze(axis=0), topelitz.t())
        result = paddle.to_tensor(extract.squeeze() % 2, dtype='int64').tolist()

        logger.info(f'Output: {result}')
        return result
