"""Statistical metrics for single collections.

These metrics can be computed on a single partition without requiring
comparison to other collections.
"""

from starlings.expressions import MetricFunction

__all__ = ["entity_count", "entropy"]

entity_count = MetricFunction("entity_count", "statistics")
r"""Measures the total number of resolved entities in a collection at a given threshold.

Entity count is a fundamental measure of resolution granularity that directly
quantifies how many distinct entities are identified in the data. As the threshold
decreases, the entity count typically decreases as more records merge together,
reflecting increased clustering aggressiveness.

For a partition $P(t) = \{E_1, E_2, ..., E_k\}$ at threshold $t$, the entity count
is simply the number of sets in the partition:

$$
\text{EntityCount}(t) = |P(t)| = k
$$

This metric helps identify the "elbow point" in threshold sweeps where the rate of
merging changes significantly.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    metrics=[sl.Metrics.stats.entity_count]
)
```
"""

entropy = MetricFunction("entropy", "statistics")
r"""Information-theoretic measure of partition uniformity.

Entropy quantifies the "disorder" or heterogeneity in entity sizes within a partition.
It measures how evenly distributed records are across entities. Lower entropy indicates
more uniform entity sizes (e.g., all entities of similar size), whilst higher entropy
suggests varied sizes. This metric is particularly useful for understanding the
structural properties of your clustering.

For a partition $P(t)$ with entities of sizes $n_1, n_2, ..., n_k$ and total $N$
records, the entropy is:

$$
H(P(t)) = -\sum_{i=1}^{k} \frac{n_i}{N} \log_2\left(\frac{n_i}{N}\right)
$$

where $\frac{n_i}{N}$ is the probability that a randomly selected record belongs to
entity $E_i$.

Key properties:
- Maximum entropy: $\log_2(N)$ when all entities are singletons
- Minimum entropy: 0 when all records belong to one entity
- Entropy decreases as clustering becomes more aggressive

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    metrics=[sl.Metrics.stats.entropy]
)
```
"""
