paddle\_quantum.backend
===============================

包含多种后端的模块。

.. py:class:: paddle_quantum.backend.Backend(value)

   基类：:py:class:`enum.Enum`

   一种枚举。

   .. py:data:: StateVector
      :value: 'state_vector'
   
   .. py:data:: DensityMatrix
      :value: 'density_matrix'
   
   .. py:data:: QuLeaf
      :value: 'quleaf'
   
   .. py:data:: UnitaryMatrix
      :value: 'unitary_matrix'

.. rubric:: Submodules

.. toctree::
   :maxdepth: 4

   paddle_quantum.backend.density_matrix
   paddle_quantum.backend.quleaf
   paddle_quantum.backend.state_vector
   paddle_quantum.backend.unitary_matrix
