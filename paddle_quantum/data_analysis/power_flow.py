# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
Power Flow model
"""

import csv
import numpy as np
import os
import warnings
import logging
from typing import Optional, List, Tuple, Callable, Union
warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import toml
from tqdm import tqdm
import paddle
from paddle_quantum.gate import *
from paddle_quantum.state import *
from paddle_quantum.linalg import *
from paddle_quantum.data_analysis.vqls import _preprocess, _postprocess, VQLS

class Bus:
    r"""
    The class of buses in power flow model.

    Args:
        data: A list of data of the bus.
        name: The name of the bus.
    
    """

    def __init__(self, data: List[float], name: str):
        self.busIndex = int(data[0]-1)
        self.type = int(data[3])
        self.voltage = float(data[4])
        self.theta = float(data[5])
        self.PLi = float(data[6])
        self.QLi = float(data[7])
        self.PGi = float(data[8])
        self.QGi = float(data[9])
        self.Qmin = float(data[12])
        self.Qmax = float(data[13])
        self.v_final = self.voltage
        self.theta_final = self.theta
        self.name = name


    @property
    def V(self) -> complex:
        r"""
        Calculate the voltage of the bus.

        Returns:
            Return the voltage of the bus.
        """
        return self.v_final * np.exp(self.theta_final * 1j)


class Branch:
    r"""
    The class of branch between different buses in power system.

    Args:
        branchIndex: The index of the branch.
        fromBus: The index of tap bus of the branch.
        toBus: The index of Z bus of the branch.
        data: A list of data of the branch.

    """
    def __init__(self, branchIndex: int, fromBus: int, toBus: int, data: List[float]):
        self.branchIndex = branchIndex
        self.fromBus = fromBus
        self.toBus = toBus
        self.r = float(data[6])
        self.x = float(data[7])
        self.b_half = float(data[8])
        self.x_prime = 1
        self.z = self.r + self.x * 1j
        self.y = 1 / self.z
        self.b = self.b_half * 1j

class Grid:
    r"""
    The class of power grid.

    Args:
        buses: The list of all buses in the grid.
        branches: The list of all braches in the grid.
        Mva_base: Mva base of power system.

    """
    def __init__(self, buses: List[Bus], branches: List[Branch], Mva_base: float):
        self.buses = buses
        self.branches = branches
        self.Mva_base = Mva_base
        self.Y = np.zeros((self.bus_num, self.bus_num), dtype=complex)
        self.branches_num = len(self.branches)
        self._Y_matrix()
        self.Pl = np.vstack([bus.PLi for bus in self.buses])
        self.Ql = np.vstack([bus.QLi for bus in self.buses])
        self.Pg = np.vstack([bus.PGi for bus in self.buses])
        self.Qg = np.vstack([bus.QGi for bus in self.buses])
        self.P_diff = self.Pg - self.Pl
        self.Q_diff = self.Qg - self.Ql


    @property
    def bus_num(self) -> int:
        r"""
        Get the number of buses in the power system.

        Returns:
            Return the number of buses.
        """
        return len(self.buses)

    def get_bus_by_number(self, number: int) -> Bus:
        r"""
        Get the bus with given number.

        Args:
            number: The bus number.

        Returns:
            Return the bus with given number.

        Raises:
            NameError: No bus with given number.
        """
        for bus in self.buses:
            if bus.busIndex == number-1:
                return bus
        raise NameError("No bus with number %d." % number)

    def get_branch_by_number(self, number: int) -> Branch:
        r"""
        Get the branch with given number.

        Args:
            number: The branch number.
            
        Returns:
            Return the branch with given number.
        
        Raises:
            NameError: No branch with given number.
        """
        for branch in self.branches:
            if branch.branchIndex == number:
                return branch
        raise NameError("No branch with number %d." % number)

    def get_branches_by_bus(self, busNumber: int) -> List[Branch]:
        r"""
        Get branches of the bus with given bus number.

        Args:
            number: The bus number.
            
        Returns:
            Return the branches of the bus with given bus number.

        """
        branches = [branch for branch in self.branches if
                 (branch.toBus.busIndex == busNumber-1 or branch.fromBus.busIndex == busNumber-1)]
        return branches

    @property
    def pq_buses(self):
        pq_buses = [bus for bus in self.buses if bus.type == 0 or bus.type==1]
        return pq_buses

    @property
    def pv_buses(self):
        pv_buses = [bus for bus in self.buses if bus.type == 2]
        return pv_buses

    def _Y_matrix(self):
        """
        Generate Y matrix.

        Returns:
            Return the Y matrix of the model.
        """
        for k in range(self.branches_num):
            branch = self.branches[k]
            fromBus = branch.fromBus.busIndex
            toBus = branch.toBus.busIndex
            self.Y[fromBus, toBus] -= branch.y/branch.x_prime
            self.Y[toBus, fromBus] = self.Y[fromBus, toBus]

        for m in range(self.bus_num):
            for n in range(self.branches_num):
                branch = self.branches[n]
                if branch.fromBus.busIndex == m:
                    self.Y[m, m] += branch.y/(branch.x_prime**2) 
                elif branch.toBus.busIndex == m:
                    self.Y[m, m] += branch.y 

    def _calculate_pf(self):
        Vt = np.vstack([bus.V for bus in self.buses]).reshape((self.bus_num, -1))
        self.I = np.matmul(self.Y, Vt)
        Iij = np.zeros((self.bus_num, self.bus_num), dtype=complex)
        Sij = np.zeros((self.bus_num, self.bus_num), dtype=complex)

        self.Im = abs(self.I)
        self.Ia = np.angle(self.I)

        for bus in self.buses:
            m = bus.busIndex  # bus index
            branches = self.get_branches_by_bus(bus.busIndex+1)
            for branch in branches:
                if branch.fromBus.busIndex == m:
                    p = branch.toBus.busIndex  # index to
                    if m != p:
                        Iij[m,p] = -(branch.fromBus.V - branch.toBus.V * branch.x_prime) * self.Y[m,p]/(branch.x_prime ** 2) + branch.b_half / (branch.x_prime ** 2) * branch.fromBus.V
                        Iij[p,m] = - (branch.toBus.V - branch.fromBus.V / branch.x_prime) * self.Y[p,m] + branch.b_half * branch.toBus.V
                else:
                    p = branch.fromBus.busIndex  # index from
                    if m != p:
                        Iij[m,p] = - (branch.toBus.V - branch.fromBus.V / branch.x_prime) * self.Y[p,m] + branch.b_half * branch.toBus.V
                        Iij[p,m] = - (branch.fromBus.V - branch.toBus.V) * self.Y[m,p] / (
                                    branch.x_prime ** 2) + branch.b_half / (branch.x_prime ** 2) * branch.fromBus.V

        self.Iij = Iij
        self.Iijr = np.real(Iij)
        self.Iiji = np.imag(Iij)

        # branch powerflows
        for m in range(self.bus_num):
            for n in range(self.bus_num):
                if n != m:
                    Sij[m,n] = self.buses[m].V * np.conj(self.Iij[m, n]) * self.Mva_base

        self.Sij = Sij
        self.Pij = np.real(Sij)
        self.Qij = np.imag(Sij)

        # branch losses
        Lij = np.zeros(self.branches_num, dtype=complex)
        for branch in self.branches:
            m = branch.branchIndex - 1
            p = branch.fromBus.busIndex
            q = branch.toBus.busIndex
            Lij[m] = Sij[p, q] + Sij[q, p]

        self.Lij = Lij
        self.Lpij = np.real(Lij)
        self.Lqij = np.imag(Lij)

        # Bus power injection
        Si = np.zeros(self.bus_num, dtype=complex)
        for i in range(self.bus_num):
            for k in range(self.bus_num):
                Si[i] += np.conj(self.buses[i].V) * self.buses[k].V * self.Y[i, k] * self.Mva_base

        self.Si = Si
        self.Pi = np.real(Si)
        self.Qi = -np.imag(Si)
        self.Pg = self.Pi.reshape([-1,1]) + self.Pl.reshape([-1,1])
        self.Qg = self.Qi.reshape([-1,1]) + self.Ql.reshape([-1,1])

    def powerflow(self, threshold, minIter: int, maxIter: int, depth: int, iterations: int, LR: float, gamma: Optional[float]=0):
        r"""
        Power flow solving process.

        Args:
            threshold: Threshold for loss value to end optmization for power flow.
            minIter: Minimum number of iterations of power flow optimization.
            maxIter: Maximum number of iteration of power flow optimization.
            depth: The depth of quantum ansatz circuit.
            iterations: Number of optimization cycles of quantum circuit.
            LR: The learning rate of the optimizer.
            gamma: Threshold for loss value to end optimization for quantum circuit early, default is 0.
        
        """
        tol=1
        self.iter = 0
        P_diff = self.P_diff / self.Mva_base
        Q_diff = self.Q_diff / self.Mva_base
        G = np.real(self.Y)
        B = np.imag(self.Y)
        angles = np.zeros((self.bus_num,1))
        pv_num = len(self.pv_buses)
        pq_num = len(self.pq_buses)
        self.tolerances = []

        logging.basicConfig(
            filename='./power_flow.log',
            filemode='w',
            format='%(asctime)s %(levelname)s %(message)s',
            level=logging.INFO
        )

        msg = f"Input parameters:"
        logging.info(msg)
        msg = f'Error threshold for power flow: {threshold}'
        logging.info(msg)
        msg = f'Minimum number of iterations of power flow: {minIter}'
        logging.info(msg)
        msg = f'Maximum number of iterations of power flow: {maxIter}'
        logging.info(msg)
        msg = f"Depth of ansatz circuit: {depth}"
        logging.info(msg)
        msg = f"Learning rate: {LR}"
        logging.info(msg)
        msg = f"Number of iterations for optimizing quantum circuits: {iterations}"
        logging.info(msg)

        if gamma == 0:
            msg = f"No threshold value was given to quantum circuit optimization."
        else:
            msg = f"Threshold value for quantum circuit optimization: {gamma}"
        logging.info(msg)

        while self.iter < minIter or (tol > threshold and self.iter < maxIter):
            self.iter += 1

            msg = f"Start iteration {self.iter} for power flow optimization"
            logging.info(msg)

            P = np.zeros((self.bus_num, 1))
            Q = np.zeros((self.bus_num, 1))

            #calculate P and Q
            for bus in self.buses:
                i = bus.busIndex
                for k in range(self.bus_num):
                    P[i] += bus.v_final*self.buses[k].v_final*(G[i, k]*np.cos(angles[i]-angles[k]) + B[i,k]*np.sin(angles[i]-angles[k]))
                    Q[i] += bus.v_final*self.buses[k].v_final*(G[i, k]*np.sin(angles[i]-angles[k]) - B[i,k]*np.cos(angles[i]-angles[k]))
            self.P = P

            #calculate gap between current injective voltage and target voltage
            dP_all = P_diff - P
            dQ_all = Q_diff - Q
            k = 0
            dQ = np.zeros((pq_num, 1))
            for bus in self.pq_buses:
                i = bus.busIndex
                if bus.type == 0 or bus.type==1:
                    dQ[k] = dQ_all[i]
                    k += 1
            dP = dP_all[1:self.bus_num]
            Mismatch = np.vstack((dP, dQ))

            # calculate Jacobian matrix. 
            # H_mat is the derivative of P with respect to angles
            H_mat = np.zeros((self.bus_num-1, self.bus_num-1))
            for i in range(self.bus_num-1):
                m = i + 1
                for k in range(self.bus_num-1):
                    n = k + 1
                    if n == m:
                        H_mat[i,k] += -Q[m]
                        H_mat[i,k] += -self.buses[m].v_final**2*B[m,m]
                    else:
                        H_mat[i,k] = self.buses[m].v_final*self.buses[n].v_final*(G[m,n]*np.sin(angles[m]-angles[n]) - B[m,n]*np.cos(angles[m]-angles[n]))
            self.H_mat = H_mat

            #N_mat is the derivative of P with respect to V
            N_mat = np.zeros((self.bus_num-1, pq_num))
            for i in range(self.bus_num-1):
                m = i + 1
                for k in range(pq_num):
                    n = self.pq_buses[k].busIndex
                    if n == m:
                        N_mat[i,k] += P[m]
                        N_mat[i,k] += self.buses[m].v_final**2*G[m,m]
                    else:
                        N_mat[i,k] = self.buses[m].v_final*self.buses[n].v_final*(G[m,n]*np.cos(angles[m]-angles[n]) + B[m,n]*np.sin(angles[m]-angles[n]))
            self.N_mat = N_mat


            #M_mat is the derivative of Q with respect to angles
            M_mat = np.zeros((pq_num, self.bus_num-1))
            for i in range(pq_num):
                m = self.pq_buses[i].busIndex
                for k in range(self.bus_num-1):
                    n = k + 1
                    if n == m:
                        M_mat[i,k] += P[m]
                        M_mat[i,k] += -self.buses[m].v_final**2*G[m,m]
                    else:
                        M_mat[i,k] = self.buses[m].v_final*self.buses[n].v_final*(-G[m,n]*np.cos(angles[m]-angles[n]) - B[m,n]*np.sin(angles[m]-angles[n]))
            self.M_mat = M_mat

            #L_mat is the derivative of Q with respect to V
            L_mat = np.zeros((pq_num, pq_num))
            for i in range(pq_num):
                m = self.pq_buses[i].busIndex
                for k in range(pq_num):
                    n = self.pq_buses[k].busIndex
                    if n == m:
                        L_mat[i,k] += Q[m]
                        L_mat[i,k] += -self.buses[m].v_final**2*B[m,m]
                    else:
                        L_mat[i,k] = self.buses[m].v_final*self.buses[n].v_final*(G[m,n]*np.sin(angles[m]-angles[n]) - B[m,n]*np.cos(angles[m]-angles[n]))
            self.L_mat = L_mat

            self.J = np.vstack((np.hstack((H_mat, N_mat)), np.hstack((M_mat, L_mat))))
            
            # Calculate the reverse of J and increment of data

  
            msg = f"Start optmization for quantum circuit"
            logging.info(msg)       
            X = compute(self.J,Mismatch[:,0], depth, iterations, LR, gamma).real
            self.X = X
            dTheta = X[0:self.bus_num-1]
            dV = X[self.bus_num-1:]
            msg = f"End optmization for quantum circuit"
            logging.info(msg) 


            #update Angles and Voltages
            #angles[0] is the angle of the slack bus

            for i in range(1,self.bus_num):
                angles[i]+= dTheta[i-1]
                
            k=0
            for i in range(1, self.bus_num):
                if self.buses[i].type == 0 or self.buses[i].type == 1:
                    self.buses[i].v_final += dV[k].item()*self.buses[i].v_final
                    k += 1
                self.buses[i].theta_final = angles[i].item()

            tol = max(abs(Mismatch))
            self.tolerances.append(tol[0])
            self.voltage = [self.buses[i].v_final for i in range(self.bus_num)]
            self.theta_final = [self.buses[i].theta_final for i in range(self.bus_num)]

            msg = f"End iteration {self.iter} for power flow optimization, current error is {tol[0]}"
            logging.info(msg)

        #calculate the power flow
        self._calculate_pf()

    def printResults(self):
        r"""
        Print the result of power flow.
        """

        print("Power Flow:")
        print()
        print('| Bus |    Bus     |    V     |  Angle   |      Injection      |      Generation     |        Load        |')
        print('| No  |    Name    |    pu    |  Degree  |     MW   |   MVar   |     MW   |  Mvar    |     MW  |     MVar |')
        for i in range(self.bus_num):
            print(f'| %3g |{self.buses[i].name}| %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f |%8.3f | %8.3f |' % (i+1, self.buses[i].v_final, self.buses[i].theta_final, self.Pi[i], self.Qi[i], self.Pg[i], self.Qg[i], self.Pl[i], self.Ql[i]))

        print('----------------------------------------------------------------------------------------------------------')
        print()
        print("Network and losses:")
        print()
        print('|  From |    To |     P     |     Q     |  From |    To |     P     |     Q     |        Branch Loss      |')
        print('|   Bus |   Bus |     MW    |    MVar   |   Bus |   Bus |     MW    |    MVar   |     MW     |     MVar   |')
        for i in range(self.branches_num):
            p = self.branches[i].fromBus.busIndex
            q = self.branches[i].toBus.busIndex
            print('| %5g | %5g | %9.2f | %9.2f | %5g | %5g | %9.2f | %9.2f |  %9.2f |  %9.2f |' % (p+1, q+1, self.Pij[p,q], self.Qij[p,q], q+1, p+1, self.Pij[q,p], self.Qij[q,p], self.Lpij[i], self.Lqij[i]))
        print('----------------------------------------------------------------------------------------------------------')
        print()
        print('Total active power losses: {active_power:.2f}, Total reactive power losses: {reactive_power:.2f}'.format(active_power=sum(self.Lpij).item(), reactive_power=sum(self.Lqij).item()))

    def saveResults(self):
        r"""
        Save the result of power flow.
        """

        with open("pf_result.txt","w") as r:
            r.write("Power Flow:")
            r.write("\n")
            r.write('| Bus |    Bus     |    V     |  Angle   |      Injection      |      Generation     |        Load        |\n')
            r.write('| No  |    Name    |    pu    |  Degree  |     MW   |   MVar   |     MW   |  Mvar    |     MW  |     MVar |\n')
            for i in range(self.bus_num):
                r.write(f'| %3g |{self.buses[i].name}| %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f |%8.3f | %8.3f |\n' % (i+1, self.buses[i].v_final, self.buses[i].theta_final, self.Pi[i], self.Qi[i], self.Pg[i], self.Qg[i], self.Pl[i], self.Ql[i]))

            r.write('----------------------------------------------------------------------------------------------------------\n')
            r.write("Network and losses:\n")
            r.write('|  From |    To |     P     |     Q     |  From |    To |     P     |     Q     |        Branch Loss      |\n')
            r.write('|   Bus |   Bus |     MW    |    MVar   |   Bus |   Bus |     MW    |    MVar   |     MW     |     MVar   |\n')
            for i in range(self.branches_num):
                p = self.branches[i].fromBus.busIndex
                q = self.branches[i].toBus.busIndex
                r.write('| %5g | %5g | %9.2f | %9.2f | %5g | %5g | %9.2f | %9.2f |  %9.2f |  %9.2f |\n' % (p+1, q+1, self.Pij[p,q], self.Qij[p,q], q+1, p+1, self.Pij[q,p], self.Qij[q,p], self.Lpij[i], self.Lqij[i]))
            r.write('---------------------------------------------------------------------------------------------------------\n')
            r.write('Total active power losses: {active_power:.2f}, Total reactive power losses: {reactive_power:.2f}\n'.format(active_power=sum(self.Lpij).item(), reactive_power=sum(self.Lqij).item()))

def compute(A: np.ndarray, b: np.ndarray, depth: int, iterations: int, LR: float, gamma: Optional[float]=0) -> np.ndarray:
    r"""
    Solve the branchar equation Ax=b.

    Args:
        A: Input matrix.
        b: Input vector.
        depth: Depth of ansatz circuit.
        iterations: Number of iterations for optimization.
        LR: Learning rate of optimizer.
        gamma: Extra option to end optimization early if loss is below this value. Default to '0'.

    Returns:
        Return the vector x that solves Ax=b.

    Raises:
        ValueError: A is not a square matrix.
        ValueError: dimension of A and b don't match.
        ValueError:  A is a singular matrix hence there's no unique solution.
            """
    # check dimension of input and invertibility of A
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not a square matrix")
    if len(A) != len(b):
        raise ValueError("dimension of A and b don't match")
    if np.linalg.det(A) == 0:
        raise ValueError("A cannot be inverted")


    b_rescale, num_qubits, list_A, coefficients_real, coefficients_img, original_dim, scale = _preprocess(A=A,b=b)
    vqls = VQLS(num_qubits=num_qubits, A=list_A, coefficients_real=coefficients_real, coefficients_img=coefficients_img, b=b_rescale, depth=depth)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=vqls.parameters())

    for itr in tqdm(range(1, iterations + 1)):
        loss = vqls()
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if itr % 10 == 0:
            msg = (
                f"Iter:{itr:5d}, Loss:{loss.item(): 3.5f}"
            )
            logging.info(msg)
        if loss.item()<gamma:
            msg='Threshold value gamma reached, ending optimization'
            logging.info(msg)
            print(msg)
            break

    estimate_state = vqls.cir(zero_state(num_qubits))
    x = _postprocess(scale=scale, original_dim=original_dim, A=A, x=estimate_state.numpy(), b=b)
    return x


def data_to_Grid(file_name) -> Grid:
    r"""
    Transfer the data file to power grid.

    Args:
        file_name: The file name of power system data.
    
    Returns:
        Return the power grid.
    """
    with open(file_name) as all_data:
        data_read = csv.reader(all_data)
        data = list(data_read)

    # Initialize a list that contain the positions of '-999'
    cuts=[]
    for i in range(len(data)):
        if data[i-1][0][:4] == '-999':
            cuts.append(i-1)        
    # Delete the last element
    del(cuts[-1]) 

    # Mva Base
    Mva_base = float(data[0][0][31:36])

    # Data preprocess
    for i in range(2,cuts[0]):
        data[i].append(data[i][0][5:17])
        data[i][0]=data[i][0][:5]+data[i][0][17:]
          
    # Extract buses data
    bus = []
    bus_name = []
    bus_num  = cuts[0] - 2 
    for i in range(2,cuts[0]):
        bus.append(data[i][0])
        bus_name.append(data[i][1])

    bus_data = np.arange(bus_num*17,dtype=float).reshape(bus_num,17)
    for i in range(bus_num):
        bus_data[i] = np.fromstring(bus[i],dtype=float,sep=' ')

      
    # Extract branches data
    branch_num = cuts[1]-cuts[0]-2
    branch = []
    for i in range(cuts[0]+2,cuts[1]):
        branch.append(data[i])
        
    branch_data = np.arange(branch_num*21,dtype=float).reshape(branch_num,21)
    for i in range(branch_num):
        branch_data[i] = np.fromstring(branch[i][0],dtype=float,sep=' ')

    # Creating a list of buses   
    bus_list = []
    for i in range(bus_num):
        n = bus_data[i]
        bus_list.append(Bus(n,name=bus_name[i]))

    # Creating a list of branches
    branch_list= []
    for i in range(branch_num):
        b = branch_data[i]
        branch_list.append(Branch(i+1,bus_list[int(b[0]-1)],bus_list[int(b[1]-1)],b)) 

    grid = Grid(buses=bus_list, branches=branch_list, Mva_base=Mva_base)
    return grid

if __name__ == '__main__':
    exit(0)