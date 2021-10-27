from paddle_quantum.circuit import UAnsatz
from paddle import kron
from paddle_quantum.state import vec,density_op
import paddle


#density_matrix
def test_density_matrix():
    cir = UAnsatz(1)
    cir.ry(paddle.to_tensor(1,dtype='float64'),0)
    state = cir.run_density_matrix()
    cir.expand(3)
    print(cir.get_state())

    cir2 = UAnsatz(3)
    cir2.ry(paddle.to_tensor(1,dtype='float64'),0)
    cir2.run_density_matrix()
    print(cir2.get_state())

#state_vector
def test_state_vector():
    cir = UAnsatz(1)
    cir.ry(paddle.to_tensor(1,dtype='float64'),0)
    state = cir.run_state_vector()
    cir.expand(3)
    print(cir.get_state())

    cir2 = UAnsatz(3)
    cir2.ry(paddle.to_tensor(1,dtype='float64'),0)
    cir2.run_state_vector()
    print(cir2.get_state())

test_density_matrix()
test_state_vector()
