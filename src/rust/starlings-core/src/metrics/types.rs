//! Common types for metric computation

/// Contingency table for comparing two partitions
#[derive(Debug, Clone)]
pub struct ContingencyTable {
    /// Number of record pairs that are in same entity in both partitions
    pub true_positives: u32,
    /// Number of record pairs that are in same entity in first but different in second
    pub false_positives: u32,
    /// Number of record pairs that are in different entities in first but same in second
    pub false_negatives: u32,
    /// Number of record pairs that are in different entities in both partitions
    pub true_negatives: u32,
}

impl ContingencyTable {
    /// Compute precision: TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            if self.false_negatives > 0 {
                0.0
            } else {
                1.0
            }
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute recall: TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            1.0
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Compute F1 score: 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }
}
