Performance Tuning
==================

Optimizing GPClarity for large datasets.

Strategies
----------

1. **Cache predictions**: Reuse profiler instances
2. **Reduce capture frequency**: tracker.wrapped_optimize(capture_every=10)
3. **Use leverage scores**: Instead of LOO for influence
4. **Parallel processing**: n_jobs parameter where available
5. **Subset analysis**: Analyze complexity on representative sample