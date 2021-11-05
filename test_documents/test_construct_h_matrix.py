from paddle_quantum.utils import Hamiltonian

h = Hamiltonian([(1, 'Z0, Z1')])

print(h.construct_h_matrix())
print(h.construct_h_matrix(4))
