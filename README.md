# Machine Learning for Protein Mutational Scanning: GFP, SUMO, and GB1

**Graduation Project — Sabancı University**

**Authors:** Mehmet Bastug & Bahadır Kaşifoğlu  
**Supervisor:** Prof. Canan Atılgan

---

## 📌 Project Overview
This repository contains the codebase for a machine learning pipeline designed to predict the functional consequences of single-point mutations in proteins. By integrating biophysical features derived from Molecular Dynamics (MD) simulations with statistical learning, we aim to map the relationship between structural changes and protein function.

We specifically focused on three distinct protein systems:
1.  **GFP (Green Fluorescent Protein):** Predicting *Median Brightness* based on chromophore dynamics.
2.  **SUMO (Small Ubiquitin-like Modifier):** Predicting binding affinity/fitness.
3.  **GB1 (Protein G B1 domain):** Predicting binding affinity/fitness.

## 🚀 Methodology

Our approach utilizes a **Random Forest Regressor** trained on a diverse set of features extracted from structural data. The pipeline is designed to be robust, handling different datasets with varying feature availability.

### Key Features
We engineered a hybrid feature set combining geometric, topological, and physical descriptors:

* **SASA (Solvent Accessible Surface Area):**
    * Whole protein SASA.
    * Local SASA (e.g., Chromophore for GFP, Chain A for SUMO/GB1).
    * $\Delta$SASA (Difference between bound/unbound or whole/local states).
* **Graph Theory Metrics:**
    * Constructed protein contact networks (using $C_\alpha$ coordinates).
    * Features: *Average Degree*, *Average Clustering Coefficient*, *Total Graph Length*.
* **Spherical Coordinates:**
    * Spatial density features calculated from atomic distributions relative to the protein center and the specific mutation site.
    * Features relative to the specific mutation site.
    * Per-residue spherical coordinates capturing local geometric context.
* **Hydrogen Bonds:**
    * Dynamic per-residue H-bond counts extracted from MD trajectories.

## 🛠️ Requirements

The pipeline requires **Python 3.9+** and the following libraries:

* `numpy`
* `pandas`
* `scikit-learn`
* `networkx` (for graph feature extraction)
* `matplotlib` & `seaborn` (for visualization)

You can install dependencies via:
```bash
pip install numpy pandas scikit-learn networkx matplotlib seaborn