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
Quantum portfolio optimization tools. 
"""
from typing import List, Union, Tuple, Optional
import logging
import warnings
import numpy as np
import paddle
import paddle_quantum as pq
from paddle_quantum.ansatz import Circuit
from paddle_quantum.finance import DataSimulator, portfolio_optimization_hamiltonian
from tqdm import tqdm


def portfolio_combination_optimization(
    num_asset: int,
    data: Union[pq.finance.DataSimulator, Tuple[paddle.Tensor, paddle.Tensor]],
    iter: int,
    lr: Optional[float] = None,
    risk: float = 0.5,
    budget: int = None,
    penalty: float = None,
    circuit: Union[pq.ansatz.Circuit, int] = 2,
    init_state: Optional[pq.state.State] = None,
    optimizer: Optional[paddle.optimizer.Optimizer] = None,
    measure_shots: int = 2048,
    logger: Optional[logging.Logger] = None,
    compare: bool = False,
) -> List[int]:
    r"""A highly encapsuled method of the portfolio combination optimization. 

    Args:
        num_asset: the number of investable asset.
        data: stock data, a `DataSimulator` or `tuple` (covariance_matrix, return_rate_vector).
        iter: number of optimization cycles.
        lr: learning rate.
        risk: the coeffcient of risk.
        budget: investment budget, or the maximum counts of projects we invest. 
        penalty: the weight of regular terms.
        circuit: the `Circuit` we use to inference. If `int` is input, we will construct `complex_entangled_layer` that times. 
        init_state: the initial state of inference circuit, default to be the product state of zero states.
        optimizer: the `paddle.optimizer.Optimizer` instance, default to be `paddle.optimizer.Adam`.
        measure_shots: the times we measure the end state, default to be 2048.
        logger: `logging.Logger` instance for detail record.
        compare: whether compare the loss of end state with the loss of ground state. This will be costly when `num_asset` too large.

    Returns:
        investment_plan: the optimal investment strategy as a list.

    Note:
        This function is only applied to a well defined problem introduced in 
        `Portfolio Optimization <https://qml.baidu.com/tutorials/combinatorial-optimization/quantum-finance-application-on-portfolio-optimization.html>`_.
    """
    if isinstance(data, pq.finance.DataSimulator):
        covar = data.get_asset_return_covariance_matrix()
        return_rate = data.get_asset_return_mean_vector()
    elif isinstance(data, Tuple):
        covar, return_rate = data
    
    if not budget:
        budget = num_asset // 2
    if not penalty:
        penalty = num_asset

    hamiltonian = portfolio_optimization_hamiltonian(penalty, return_rate, covar, risk, budget)
    loss_func = pq.loss.ExpecVal(hamiltonian)
    num_qubits = num_asset
    if not init_state:
        init_state = pq.state.zero_state(num_qubits)

    if isinstance(circuit, int):
        depth = circuit
        circuit = Circuit(num_qubits)
        circuit.complex_entangled_layer(depth=depth)
    
    if not optimizer:
        opt = paddle.optimizer.Adam(learning_rate=lr, parameters=circuit.parameters())
    
    for itr in tqdm(range(iter)):
        state = circuit(init_state) 
        loss = loss_func(state)
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if logger:
            logger.info(f'iters: {itr}    loss: {float(loss):.7f}')

    final_state = circuit(init_state)
    prob_measure = final_state.measure(shots = measure_shots)
    investment = max(prob_measure, key = prob_measure.get)
    investment_plan = [enum + 1 for enum, flag in enumerate(list(investment)) if flag == '1']

    if compare:
        if not logger:
            raise RuntimeError('if compared, logger must exist')
        if num_qubits > 12:
            warnings.warn('comparison beyond 12 qubits will cost unexpected time.')
        hc_mat = hamiltonian.construct_h_matrix()
        logger.info(f'the loss of ground state: {float(np.linalg.eigvalsh(hc_mat)[0]):.7f}')
        logger.info(f'the loss of ours: {float(loss):.7f}')

    return investment_plan
