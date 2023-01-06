# !/usr/bin/env python3
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
Functions and data simulator class of quantum finance.
"""

from datetime import datetime
from typing import Optional
import fastdtw
import networkx
import numpy as np
import paddle_quantum

__all__ = [
    "DataSimulator",
    "portfolio_optimization_hamiltonian",
    "portfolio_diversification_hamiltonian",
    "arbitrage_opportunities_hamiltonian"
]


class DataSimulator:
    r"""Used to generate data and calculate relevant parameters for portfolio optimization and portfolio diversification problems.
    
    Args:
        stocks: A list of names of investable stocks
        start: The start date of the trading day when the stock data is randomly generated. Defaults to ``None``.
        end: The end date of the trading day when the stock data is randomly generated. Defaults to ``None``.
    """
    def __init__(self, stocks: list, start: Optional[datetime] = None, end: Optional[datetime] = None):
        self._n = len(stocks)
        self._stocks = stocks

        if start and end:
            self._start = start
            self._end = end

        self._data = []

        self.asset_return_mean = None
        self.asset_return_cov = None

    def set_data(self, data: list) -> None:
        r"""Decide which data source to use: randomly generated or locally entered.

        Args:
            data: Stock data entered by the user.
        """
        if len(data) == self._n:
            self._data = data
        else:
            print("invalid data, data is still empty.")
            self._data = []

    def randomly_generate(self) -> None:
        r"""Randomly generate stock data for experiments based on start date and end date.

        Note:
            To generate random stock data, you need to specify the start date and end date in the format of the ``datetime`` package, e.g. ``start = datetime.datetime(2016, 1, 1)``.
        """

        if self._start and self._end:
            num_days = (self._end - self._start).days
            for _ in self._stocks:
                fluctuation = np.random.standard_normal(num_days)
                fluctuation = np.cumsum(fluctuation)
                data_i = np.random.randint(1, 101, size=1) + fluctuation
                trimmed_data_i = [max(data_i[j], 0) for j in range(num_days)]
                if 0 in trimmed_data_i:
                    zero_ind = trimmed_data_i.index(0)
                    trimmed_data_i = trimmed_data_i[:zero_ind] + [0 for _ in range(num_days - zero_ind)]

                self._data.append(trimmed_data_i)
        else:
            print("Please provide the start time and the end time you want to generate stock data.")

    def get_asset_return_mean_vector(self) -> list:
        r"""Calculate expected return of each stock.

        Returns:
            Expected return of all investable stocks.
        """
        returns = []
        for i in range(self._n):
            return_i = [self._data[i][j + 1] / self._data[i][j] - 1
                        if self._data[i][j] != 0
                        else np.nan for j in range(len(self._data[i]) - 1)]
            returns.append(return_i)
        self.asset_return_mean = np.mean(returns, axis=1)

        return self.asset_return_mean

    def get_asset_return_covariance_matrix(self) -> list:
        r"""Calculate the covariance matrix between the returns of each stock.

        Returns:
            The covariance matrix between the returns of each stock.
        """
        returns = []
        for i in range(self._n):
            return_i = [self._data[i][j + 1] / self._data[i][j] - 1
                        if self._data[i][j] != 0
                        else np.nan for j in range(len(self._data[i]) - 1)]
            returns.append(return_i)
        self.asset_return_cov = np.cov(returns)

        return self.asset_return_cov

    def get_similarity_matrix(self) -> list:
        r"""Calculate the similarity matrix among stocks.

        The Dynamic Time Warping algorithm (DTW) is used to calculate the similarity between two stocks.

        Returns:
            The similarity matrix among stocks.
        """
        rho = np.zeros((self._n, self._n))
        for i in range(0, self._n):
            rho[i, i] = 1
            for j in range(i + 1, self._n):
                curr_rho, _ = fastdtw.fastdtw(self._data[i], self._data[j])
                curr_rho = 1 / curr_rho
                rho[i, j] = curr_rho
                rho[j, i] = curr_rho

        return rho


def portfolio_optimization_hamiltonian(penalty: int, mu: list, sigma: list, q: float, budget: int) -> paddle_quantum.Hamiltonian:
    r"""Construct the hamiltonian of the portfolio optimization problem.

    Args:
        penalty: Penalty parameter.
        mu: Expected return of each stock.
        sigma: The covariance matrix between the returns of each stock.
        q: Risk appetite of the decision maker.
        budget:  Budget, i.e. the number of stocks to be invested.

    .. math::

        C(x) = q \sum_i \sum_j S_{ji}x_ix_j  - \sum_{i}x_i \mu_i + A \left(B - \sum_i x_i\right)^2


    Hint:
        Mapping Boolean variables :math:`x_i` to Hamiltonian matrices under :math:`x_i \mapsto \frac{I-Z_i}{2}`.

    Returns:
        The hamiltonian of the portfolio optimization problem.
    """
    n = len(mu)

    h_c_list1 = []
    for i in range(0, n):
        for j in range(0, n):
            sigma_ij = sigma[i][j]
            h_c_list1.append([sigma_ij / 4, 'I'])
            if i != j:
                h_c_list1.append([sigma_ij / 4, 'Z' + str(i) + ',Z' + str(j)])
            else:
                h_c_list1.append([sigma_ij / 4, 'I'])
            h_c_list1.append([- sigma_ij / 4, 'Z' + str(i)])
            h_c_list1.append([- sigma_ij / 4, 'Z' + str(j)])
    h_c_list1 = [[q * c, s] for (c, s) in h_c_list1]

    h_c_list2 = []
    for i in range(0, n):
        h_c_list2.append([- mu[i] / 2, 'I'])
        h_c_list2.append([mu[i] / 2, 'Z' + str(i)])

    h_c_list3 = [[budget ** 2, 'I']]
    for i in range(0, n):
        h_c_list3.append([- 2 * budget / 2, 'I'])
        h_c_list3.append([2 * budget / 2, 'Z' + str(i)])
        h_c_list3.append([2 / 4, 'I'])
        h_c_list3.append([- 2 / 4, 'Z' + str(i)])
        for ii in range(0, i):
            h_c_list3.append([2 / 4, 'I'])
            h_c_list3.append([2 / 4, 'Z' + str(i) + ',Z' + str(ii)])
            h_c_list3.append([- 2 / 4, 'Z' + str(i)])
            h_c_list3.append([- 2 / 4, 'Z' + str(ii)])
    h_c_list3 = [[penalty * c, s] for (c, s) in h_c_list3]

    h_c_list = h_c_list1 + h_c_list2 + h_c_list3
    po_hamiltonian = paddle_quantum.Hamiltonian(h_c_list)

    return po_hamiltonian


def portfolio_diversification_hamiltonian(penalty: int, rho: list, q: int) -> paddle_quantum.Hamiltonian:
    r"""Construct the hamiltonian of the portfolio diversification problem.

    Args:
        penalty: Penalty parameter.
        rho: The similarity matrix among stocks.
        q: Number of categories for stock clustering.

    .. math::

        \begin{aligned}
        C_x &= -\sum_{i=1}^{n}\sum_{j=1}^{n}\rho_{ij}x_{ij} + A\left(q- \sum_{j=1}^n y_j \right)^2 + \sum_{i=1}^n A\left(\sum_{j=1}^n 1- x_{ij} \right)^2 \\
            &\quad + \sum_{j=1}^n A\left(x_{jj} - y_j\right)^2 + \sum_{i=1}^n \sum_{j=1}^n A\left(x_{ij}(1 - y_j)\right).\\
        \end{aligned}

    Hint:
        Mapping Boolean variables :math:`x_{ij}` to the Hamiltonian matrices under :math:`x_{ij} \mapsto \frac{I-Z_{ij}}{2}`

    Returns:
        The hamiltonian of the portfolio diversification problem.
    """
    n = len(rho)

    h_c_list1 = []
    for i in range(0, n):
        for j in range(0, n):
            rho_ij = - rho[i][j]
            h_c_list1.append([rho_ij / 2, 'I'])
            h_c_list1.append([- rho_ij / 2, 'Z' + str(i * n + j)])

    h_c_list2 = [[q ** 2, 'I']]
    for j in range(0, n):
        h_c_list2.append([- q, 'I'])
        h_c_list2.append([q, 'Z' + str(n ** 2 + j)])
        h_c_list2.append([1 / 2, 'I'])
        h_c_list2.append([- 1 / 2, 'Z' + str(n ** 2 + j)])
        for jj in range(j):
            h_c_list2.append([1 / 2, 'I'])
            h_c_list2.append([1 / 2, 'Z' + str(n ** 2 + j) + ',Z' + str(n ** 2 + jj)])
            h_c_list2.append([- 1 / 2, 'Z' + str(n ** 2 + j)])
            h_c_list2.append([- 1 / 2, 'Z' + str(n ** 2 + jj)])
    h_c_list2 = [[penalty * c, s] for (c, s) in h_c_list2]

    h_c_list3 = []
    for i in range(0, n):
        h_c_list3.append([1, 'I'])
        for j in range(0, n):
            h_c_list3.append([- 1, 'I'])
            h_c_list3.append([1, 'Z' + str(i * n + j)])
            h_c_list3.append([1 / 2, 'I'])
            h_c_list3.append([- 1 / 2, 'Z' + str(i * n + j)])
            for jj in range(j):
                h_c_list3.append([1 / 2, 'I'])
                h_c_list3.append([- 1 / 2, 'Z' + str(i * n + j)])
                h_c_list3.append([1 / 2, 'Z' + str(i * n + j) + ',Z' + str(i * n + jj)])
                h_c_list3.append([- 1 / 2, 'Z' + str(i * n + jj)])
    h_c_list3 = [[penalty * c, s] for (c, s) in h_c_list3]

    h_c_list4 = []
    for j in range(0, n):
        h_c_list4.append([1 / 2, 'I'])
        h_c_list4.append([- 1 / 2, 'Z' + str(j * n + j) + ',Z' + str(n ** 2 + j)])
    h_c_list4 = [[penalty * c, s] for (c, s) in h_c_list4]

    h_c_list5 = []
    for i in range(0, n):
        for j in range(0, n):
            h_c_list5.append([1 / 4, 'I'])
            h_c_list5.append([- 1 / 4, 'Z' + str(i * n + j)])
            h_c_list5.append([1 / 4, 'Z' + str(n ** 2 + j)])
            h_c_list5.append([- 1 / 4, 'Z' + str(i * n + j) + ',Z' + str(n ** 2 + j)])
    h_c_list5 = [[penalty * c, s] for (c, s) in h_c_list5]

    h_c_list = h_c_list1 + h_c_list2 + h_c_list3 + h_c_list4 + h_c_list5
    pd_hamiltonian = paddle_quantum.Hamiltonian(h_c_list)

    return pd_hamiltonian


def arbitrage_opportunities_hamiltonian(g: networkx.DiGraph, penalty: int, n: int, k: int) -> paddle_quantum.Hamiltonian:
    r"""Construct the hamiltonian of the arbitrage opportunity optimization problem.

    Args:
        g: Graphical representation of conversions between different markets.
        penalty: Penalty parameter.
        n: Number of currency types, i.e. number of vertices in the graph g.
        k: Number of vertices contained in the arbitrage loop.

    .. math::

        C(x) = - P(x) + A\sum_{k=0}^{K-1} \left(1 - \sum_{i=0}^{n-1} x_{i,k}\right)^2 + A\sum_{k=0}^{K-1}\sum_{(i,j)\notin E}x_{i,k}x_{j,k+1}

    Hint:
        Mapping Boolean variables :math:`x_{i,k}` to the Hamiltonian matrices under :math:`x_{i,k} \mapsto \frac{I-Z_{i,k}}{2}`.

    Returns:
        The hamiltonian of the arbitrage opportunity optimization problem.
    """
    nodes = list(g.nodes)

    h_c_list1 = []
    for i, c in enumerate(nodes):
        for j, cc in enumerate(nodes):
            if i != j:
                c_ij = np.log2(g[c][cc]['weight'])
                for t in range(k):
                    h_c_list1.append([c_ij / 4, 'I'])
                    h_c_list1.append([c_ij / 4, 'Z' + str(i * n + t) + ',Z' + str((j * n + (t + 1) % k))])
                    h_c_list1.append([- c_ij / 4, 'Z' + str(i * n + t)])
                    h_c_list1.append([- c_ij / 4, 'Z' + str((j * n + (t + 1) % k))])
    h_c_list1 = [[-c, s] for (c, s) in h_c_list1]

    h_c_list2 = []
    for t in range(k):
        h_c_list2.append([1, 'I'])
        for i in range(n):
            h_c_list2.append([- 2 * 1 / 2, 'I'])
            h_c_list2.append([2 * 1 / 2, 'Z' + str(i * n + t)])
            h_c_list2.append([2 / 4, 'I'])
            h_c_list2.append([- 2 / 4, 'Z' + str(i * n + t)])
            for ii in range(i):
                h_c_list2.append([2 / 4, 'I'])
                h_c_list2.append([2 / 4, 'Z' + str(i * n + t) + ',Z' + str(ii * n + t)])
                h_c_list2.append([- 2 / 4, 'Z' + str(i * n + t)])
                h_c_list2.append([- 2 / 4, 'Z' + str(ii * n + t)])
    h_c_list2 = [[penalty * c, s] for (c, s) in h_c_list2]

    h_c_list3 = []
    for t in range(k):
        for (i, c) in enumerate(nodes):
            for (j, cc) in enumerate(nodes):
                if (c, cc) not in g.edges and c != cc:
                    h_c_list3.append([1 / 4, "I"])
                    h_c_list3.append([- 1 / 4, 'Z' + str(i * n + t)])
                    h_c_list3.append([- 1 / 4, 'Z' + str((j * n + (t + 1) % k))])
                    h_c_list3.append([- 1 / 4, 'Z' + str(i * n + t) + ',Z' + str((j * n + (t + 1) % k))])
    h_c_list3 = [[penalty * c, s] for (c, s) in h_c_list3]

    h_c_list4 = []
    for i in range(n):
        h_c_list4.append([1, 'I'])
        for t in range(k):
            h_c_list4.append([- 2 * 1 / 2, 'I'])
            h_c_list4.append([2 * 1 / 2, 'Z' + str(i * n + t)])
            h_c_list4.append([2 / 4, 'I'])
            h_c_list4.append([- 2 / 4, 'Z' + str(i * n + t)])
            for tt in range(t):
                h_c_list4.append([2 / 4, 'I'])
                h_c_list4.append([2 / 4, 'Z' + str(i * n + t) + ',Z' + str(i * n + tt)])
                h_c_list4.append([- 2 / 4, 'Z' + str(i * n + t)])
                h_c_list4.append([- 2 / 4, 'Z' + str(i * n + tt)])
    h_c_list4 = [[penalty * c, s] for (c, s) in h_c_list4]

    h_c_list = h_c_list1 + h_c_list2 + h_c_list3 + h_c_list4
    ao_hamiltonian = paddle_quantum.Hamiltonian(h_c_list)

    return ao_hamiltonian
