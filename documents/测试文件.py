from paddle_quantum.state import vec_random,density_op_random
from paddle_quantum.utils import plot_density_graph


state = paddle.to_tensor(vec_random(2),dtype='float64')
density_matrix = density_op_random(2)

plot_density_graph(state)
plot_density_graph(density_matrix)
