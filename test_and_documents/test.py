import math
import numpy as np
import paddle
from paddle import matmul, transpose, trace
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger, random_pauli_str_generator, pauli_str_to_matrix
from paddle_quantum.state import vec, vec_random, density_op, density_op_random
theta_size = 4
num_qubits = 2
ITR = 80
LR=0.5
SEED = 1
paddle.seed(SEED)
path = '/home/aistudio/飞桨常规赛：量子电路合成/Question_2_Unitary.txt'


#普通构建方法
def U_theta(theta,num_qubits):
    cir = UAnsatz(num_qubits)
    cir.ry(theta[0],0)
    cir.ry(theta[1],1)
    cir.cnot([0,1])
    cir.ry(theta[2],0)
    cir.ry(theta[3],1)
    return cir.U

#通过量子比特扩展
def U_theta2(theta,num_qubits):
    cir = UAnsatz(1)
    cir.ry(theta[0],0)
    cir.expand(num_qubits)
    cir.ry(theta[1],1)
    cir.cnot([0,1])
    cir.ry(theta[2],0)
    cir.ry(theta[3],1)
    return cir.U

class Optimization_exl(paddle.nn.Layer):
    def __init__(self,shape,dtype='float64'):
        super(Optimization_exl,self).__init__()
        f = np.loadtxt(path)
        self.u = paddle.to_tensor(f)
        self.theta = self.create_parameter(shape=shape,
                                            default_initializer = paddle.nn.initializer.Uniform(low=0,high=2*np.pi),
                                            dtype=dtype,is_bias=False)
    
    def forward(self):
        #U = U_theta(self.theta,num_qubits)
        U = U_theta2(self.theta,num_qubits)
        U_dagger = dagger(U)
        loss = 1-(0.25*paddle.real(trace(matmul(self.u,U_dagger)))[0])
        return loss
    
    def result(self):
        return self.theta

loss_list = []
parameter_list = []
myLayer = Optimization_exl([theta_size])
opt = paddle.optimizer.Adam(learning_rate=LR,parameters=myLayer.parameters())

for itr in range(ITR):
    loss = myLayer()[0]
    loss.backward()
    opt.minimize(loss)
    opt.clear_grad()

    loss_list.append(loss.numpy()[0])
    parameter_list.append(myLayer.parameters()[0].numpy())
    #if itr % 5 == 0:
        #print('iter:', itr, '  loss: %.4f' % loss.numpy())
print(myLayer.result())
"""
U_theta 结果
Tensor(shape=[4], dtype=float64, place=CPUPlace, stop_gradient=False,
       [-0.43797367,  7.05269038,  3.48305136,  0.42135362])
U_theta2 结果
Tensor(shape=[4], dtype=float64, place=CPUPlace, stop_gradient=False,
       [-0.43797367,  7.05269038,  3.48305136,  0.42135362])
"""
