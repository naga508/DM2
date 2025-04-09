# Assignment 2: K-Means and Hierarchical Clustering

## Contents

This submission includes solutions to all four parts of the clustering assignment, along with required `.pkl` output files and plots.

### 📂 Files

#### 📍 Part 1: K-Means on Diverse Datasets
- `part1.py`: Code to evaluate k-means on synthetic datasets.
- `part1.pkl`: Saved results dictionary.
- `part1_clusters.pdf`: Visualizations of k-means clustering.

#### 📍 Part 2: Clustering Evaluation Metrics
- `part2_superclean.py`: Implements SSE and inertia comparisons for k-means.
- `part2_superclean.pkl`: Final output with function references and plots.
  
#### 📍 Part 3: Hierarchical Clustering on Toy Dataset
- `part3_superclean.py`: Hierarchical clustering on provided `.mat` data.
- `part3_superclean.pkl`: Output dictionary matching test structure.

#### 📍 Part 4: Hierarchical Clustering on Real Datasets
- `part4.py`: Implements linkage-based clustering and dendrogram thresholding.
- `part4.pkl`: Final dictionary of clustering results.
- `part4_hierarchical_linkages.pdf`: Clustering visualized by linkage type.
- `part4_thresholds.pdf`: Visualizations of threshold-based cuts.

#### 📍 Reference
- `datamining_assignment_2_s2025_from_gz.pdf`: Assignment PDF.

## Notes

- All functions follow the required signatures.
- All `.pkl` files pass structural autograder checks.
- Plots were generated using `matplotlib`.

## How to Run

Each `part*.py` file can be executed individually:
```bash
python part1.py
python part2_superclean.py
python part3_superclean.py
python part4.py
```
Each will generate its own `.pkl` and PDF outputs.

---

### ✅ Ready to Submit
This archive includes everything needed for grading.