"""Evaluation metrics for comparing partitions.

These metrics require two or more collections to compare partitions
and compute agreement measures.
"""

from starlings.expressions import MetricFunction

__all__ = [
    "precision",
    "recall",
    "f1",
    "bcubed_precision",
    "bcubed_recall",
    "ari",
    "nmi",
    "v_measure",
]

# Asymmetric metrics (require reference)

precision = MetricFunction("precision", "evaluation")
r"""Measures the proportion of predicted entity pairs that are correct.

Asymmetric. Requires a reference collection.

Precision quantifies how many of the pairs that the system predicted to be in the
same entity actually belong together according to the reference. High precision means
few false positives - the system rarely groups records that should be separate.

For partitions $P$ (predicted) and $R$ (reference), precision is calculated using
the contingency table:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

where TP (true positives) are pairs correctly identified as belonging together,
and FP (false positives) are pairs incorrectly grouped together.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95).reference(),
    metrics=[sl.Metrics.eval.precision]
)
```
"""

recall = MetricFunction("recall", "evaluation")
r"""Measures the proportion of true entity pairs that are found.

Asymmetric. Requires a reference collection.

Recall quantifies how many of the pairs that actually belong together (according to
the reference) were correctly identified by the system. High recall means few false
negatives - the system rarely misses records that should be grouped together.

For partitions $P$ (predicted) and $R$ (reference), recall is calculated using the
contingency table:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

where TP (true positives) are pairs correctly identified as belonging together, and
FN (false negatives) are pairs that should be together but were separated.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95).reference(),
    metrics=[sl.Metrics.eval.recall]
)
```
"""

f1 = MetricFunction("f1", "evaluation")
r"""Harmonic mean of precision and recall.

Asymmetric. Requires a reference collection.

F1 score balances precision and recall, providing a single measure of clustering
quality that considers both false positives and false negatives. It reaches its best
value at 1 (perfect precision and recall) and worst at 0. The harmonic mean ensures
that both precision and recall must be high for a good F1 score.

For precision $P$ and recall $R$:

$$
\text{F1} = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

The F1 score is the harmonic mean rather than arithmetic mean because it penalises
extreme values - a model with perfect precision but zero recall (or vice versa) will
have F1 = 0.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95).reference(),
    metrics=[sl.Metrics.eval.f1]
)
```
"""


# B-cubed metrics

bcubed_precision = MetricFunction("bcubed_precision", "evaluation")
r"""B-cubed precision: entity-centric precision measure.

Asymmetric. Requires a reference collection.

B-cubed precision evaluates clustering quality from the perspective of individual
records rather than pairs. For each record, it measures the proportion of records in
its predicted cluster that should be there according to the reference, then averages
across all records. This provides a more nuanced view than pairwise precision.

For each record $r$, let $C(r)$ be its predicted cluster and $L(r)$ be its true
cluster. B-cubed precision for $r$ is:

$$
\text{Precision}(r) = \frac{|C(r) \cap L(r)|}{|C(r)|}
$$

The overall B-cubed precision is:

$$
\text{B}^3\text{-Precision} = \frac{1}{n} \sum_{r=1}^{n} \text{Precision}(r)
$$

This metric gives equal weight to each record, making it sensitive to how well
individual records are clustered.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95).reference(),
    metrics=[sl.Metrics.eval.bcubed_precision]
)
```
"""

bcubed_recall = MetricFunction("bcubed_recall", "evaluation")
r"""B-cubed recall: entity-centric recall measure.

Asymmetric. Requires a reference collection.

B-cubed recall evaluates clustering quality from the perspective of individual
records rather than pairs. For each record, it measures the proportion of records
that should be in its cluster (according to the reference) that are actually there,
then averages across all records. This provides a more nuanced view than pairwise
recall.

For each record $r$, let $C(r)$ be its predicted cluster and $L(r)$ be its true
cluster. B-cubed recall for $r$ is:

$$
\text{Recall}(r) = \frac{|C(r) \cap L(r)|}{|L(r)|}
$$

The overall B-cubed recall is:

$$
\text{B}^3\text{-Recall} = \frac{1}{n} \sum_{r=1}^{n} \text{Recall}(r)
$$

This metric gives equal weight to each record, making it sensitive to how completely
true clusters are recovered.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95).reference(),
    metrics=[sl.Metrics.eval.bcubed_recall]
)
```
"""

# Symmetric metrics

ari = MetricFunction("ari", "evaluation")
r"""Adjusted Rand Index for chance-corrected clustering agreement.

ARI measures the similarity between two partitions, adjusted for the agreement
expected by chance. It considers all pairs of records and counts pairs that are
assigned consistently or inconsistently across partitions. Values range from -1
(worse than random) through 0 (random clustering) to 1 (perfect agreement).

For two partitions $U = \{U_1, ..., U_r\}$ and $V = \{V_1, ..., V_c\}$, let
$n_{ij}$ be the number of records in both $U_i$ and $V_j$:

$$
\text{ARI} = \frac{\text{Index} - \text{Expected}}{\text{Max} - \text{Expected}}
$$

where:

- Index = $\sum_{ij} \binom{n_{ij}}{2}$ (observed co-occurrences)
- Expected = $\frac{\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}}{\binom{n}{2}}$ (chance)
- Max = $\frac{1}{2}[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}]$ (maximum)
- $a_i = \sum_j n_{ij}$, $b_j = \sum_i n_{ij}$, $n$ = total records

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95),
    metrics=[sl.Metrics.eval.ari]
)
```
"""

nmi = MetricFunction("nmi", "evaluation")
r"""Normalised Mutual Information between partitions.

NMI quantifies the mutual dependence between two partitions using information theory.
It measures how much knowing one partition reduces uncertainty about the other,
normalised by the average entropy of both partitions. Values range from 0
(independent partitions) to 1 (perfect mutual information).

For two partitions $U$ and $V$, NMI is defined as:

$$
\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}
$$

where $I(U; V)$ is the mutual information:

$$
I(U; V) = \sum_{i,j} \frac{n_{ij}}{n} \log\left(
    \frac{n \cdot n_{ij}}{a_i \cdot b_j}
\right)
$$

and $H(U)$ and $H(V)$ are the entropies of the partitions. Here $n_{ij}$ is the
number of records in both cluster $i$ of $U$ and cluster $j$ of $V$.

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95),
    metrics=[sl.Metrics.eval.nmi]
)
```
"""

v_measure = MetricFunction("v_measure", "evaluation")
r"""V-measure: harmonic mean of homogeneity and completeness.

V-measure balances two desirable properties of clusterings. Homogeneity ensures each
cluster contains only members of a single class, whilst completeness ensures all
members of a class are assigned to the same cluster. V-measure is their harmonic
mean, providing a single score that requires both properties to be satisfied.

For partitions $C$ (clusters) and $K$ (classes):

$$
\text{V-measure} = 
    2 \cdot \frac{
        \text{homogeneity} \cdot \text{completeness}
    }{
        \text{homogeneity} + \text{completeness}
    }
$$

where:

$$
\text{homogeneity} = 1 - \frac{H(K|C)}{H(K)}
$$

$$
\text{completeness} = 1 - \frac{H(C|K)}{H(C)}
$$

$H(K|C)$ is the conditional entropy of classes given clusters, and $H(K)$ is the
entropy of classes. V-measure ranges from 0 (worst) to 1 (perfect clustering).

```python
ef.analyse(
    sl.col("predicted").sweep(0.5, 1.0, 0.05),
    sl.col("ground_truth").at(0.95),
    metrics=[sl.Metrics.eval.v_measure]
)
```
"""
