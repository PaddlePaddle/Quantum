- 通过在UAnsatz类中添加新的成员函数expand来实现扩展


- 增加utils.plot_density_graph密度矩阵可视化工具。 
```
Args:
density_matrix (numpy.ndarray or paddle.Tensor): 多量子比特的量子态的状态向量或者密度矩阵,要求量子数大于1
size (float): 条宽度，在0到1之间，默认0.3
Returns:
plt.Figure: 对应的密度矩阵可视化3D直方图
```