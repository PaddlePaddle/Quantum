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
main
"""
import numpy as np
import networkx as nx
import paddle
from paddle_quantum.QAOA.QAOA_Prefunc import Generate_H_D, Draw_cut_graph, Draw_original_graph, Generate_default_graph
from paddle_quantum.QAOA.Paddle_QAOA import Paddle_QAOA

SEED = 1024

def main(n = 4):
    paddle.seed(SEED)
    
    p = 4 # number of layers in the circuit  
    ITR = 120  #number of iterations
    LR = 0.1    #learning rate

    G, V, E = Generate_default_graph(n)
    G.add_nodes_from(V)
    G.add_edges_from(E)
    
    Draw_original_graph(G)
    
    #construct the Hamiltonia
    H_D_list, H_D_matrix = Generate_H_D(E, n)
    H_D_diag = np.diag(H_D_matrix).real
    H_max = np.max(H_D_diag)

    print(H_D_diag)
    print('H_max:', H_max)   

    cir, _, _ = Paddle_QAOA(n, p, E, V, H_D_list, ITR, LR)
    prob_measure = cir.measure(plot=True)
    cut_bitstring = max(prob_measure, key=prob_measure.get)
    print("找到的割的比特串形式：", cut_bitstring)

    Draw_cut_graph(V, E, G, cut_bitstring)


if __name__ == "__main__":
    main()
