from paddle_quantum.circuit import UAnsatz
import matplotlib.pyplot as plt
from paddle_quantum.utils import plot_density_graph
import numpy as np
import paddle
import unittest


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


class TestPlotDensityGraph(unittest.TestCase):
    def setUp(self):
        self.func = plot_density_graph
        self.x_np = (np.random.rand(8, 8) + np.random.rand(8, 8) * 1j)-0.5-0.5j
        self.x_tensor = paddle.to_tensor(self.x_np)

    def test_input_type(self):
        self.assertRaises(TypeError, self.func, 1)
        self.assertRaises(TypeError, self.func, [1, 2, 3])

    def test_input_shape(self):
        x = np.zeros((2, 3))
        self.assertRaises(ValueError, self.func, x)

    def test_ndarray_input_inputs(self):
        res = self.func(self.x_np)
        res.show()

    def test_tensor_input(self):
        res = self.func(self.x_tensor)
        res.show()


if __name__ == '__main__':
    test_density_matrix()
    test_state_vector()
    unittest.main()