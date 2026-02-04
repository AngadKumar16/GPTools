Scikit-Learn Integration
========================

Using GPClarity with sklearn GaussianProcessRegressor.

.. code-block:: python

   from sklearn.gaussian_process import GaussianProcessRegressor
   
   sklearn_gp = GaussianProcessRegressor()
   sklearn_gp.fit(X, y)
   
   # Wrap for GPClarity (limited functionality)
   from gpclarity.utils import SklearnGPWrapper
   wrapped = SklearnGPWrapper(sklearn_gp)
   
   summary = gpclarity.summarize_kernel(wrapped)