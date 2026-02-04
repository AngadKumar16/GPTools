Custom Complexity Scorers
=========================

Creating your own complexity scoring strategy.

Example
-------

.. code-block:: python

   from gpclarity.model_complexity import ComplexityScorer
   
   def my_scorer(model, X, **kwargs):
       # Your custom logic
       return complexity_score
   
   analyzer = ComplexityAnalyzer(scoring_strategy=my_scorer)