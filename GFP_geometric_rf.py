import os
import re
import glob
import time
import math
import argparse

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr


# =============================================================================
# Global config
# =============================================================================

SHOW_PLOTS = False   # Set True via CLI to display plots (local runs)
OUTPUT_DIR = "results"


def _ensure_output_dir():
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_and_maybe_show(fig, filename):
    _ensure_output_dir()
    if filename:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        fig.show()
    plt.close(fig)


# =============================================================================
# Plotting functions (English, HPC-friendly)
# =============================================================================

def plot_predictions(y_true, y_pred, filename="predictions_scatter.png"):
    """
    Simple scatter plot: actual vs predicted.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", label="Predicted vs Actual")
    lo, hi = np.min(y_true), np.max(y_true)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, label="Perfect (y=x)")
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Actual vs Predicted")
    ax.grid(True)
    ax.legend()
    _save_and_maybe_show(fig, filename)


def plot_predictions_fixed(y_true, y_pred, filename="predictions_fixed_0_4.png"):
    """
    Scatter plot focused on 0–4 range, with RMSD and Pearson annotations.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = np.abs(y_pred - y_true)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sc = ax.scatter(y_true, y_pred, c=errors, alpha=0.7, edgecolors='k')
    fig.colorbar(sc, ax=ax, label="|Actual - Predicted|")

    rng = [0, 4]
    ax.plot(rng, rng, 'k-', lw=2, label="y = x")

    x_fill = np.array(rng)
    ax.fill_between(x_fill, x_fill - 0.5, x_fill + 0.5, alpha=0.1, label='±0.5')
    ax.fill_between(x_fill, x_fill - 0.25, x_fill + 0.25, alpha=0.2, label='±0.25')

    rmsd = math.sqrt(mean_squared_error(y_true, y_pred))
    pear = pearsonr(y_true, y_pred)[0]
    txt = f'RMSD: {rmsd:.3f}\nPearson: {pear:.3f}'
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), va='top')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(rng)
    ax.set_ylim(rng)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    _save_and_maybe_show(fig, filename)


def plot_predictions_with_confidence(y_true, mean_preds, ci_low, ci_high,
                                     title_suffix="Test",
                                     filename_prefix="predictions_with_ci"):
    """
    Error-bar plot with 95% CI per data point in 0–4 range.
    """
    y_true = np.asarray(y_true)
    mean_preds = np.asarray(mean_preds)
    ci_low = np.asarray(ci_low)
    ci_high = np.asarray(ci_high)
    y_err = np.vstack([mean_preds - ci_low, ci_high - mean_preds])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.errorbar(y_true, mean_preds, yerr=y_err, fmt='o', alpha=0.6,
                markersize=4, capsize=3, elinewidth=1)
    rng = [0, 4]
    ax.plot(rng, rng, 'r--', lw=2, label="y = x")
    ax.set_xlim(rng)
    ax.set_ylim(rng)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Bootstrap mean prediction (95% CI)")
    ax.set_title(f"Actual vs Mean Prediction (CI) — {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    filename = f"{filename_prefix}_{title_suffix.replace(' ', '_')}.png"
    _save_and_maybe_show(fig, filename)


def plot_bootstrapped_metric_bars(metric_means, metric_cis, metric_names,
                                  filename="bootstrap_metric_bars.png"):
    """
    Bar plot summarizing bootstrapped metrics with 95% CI.
    metric_means: list/array of length k
    metric_cis: list/array of shape (k, 2) with [low, high] per metric
    """
    metric_means = np.asarray(metric_means)
    metric_cis = np.asarray(metric_cis)
    k = len(metric_names)

    y_err = np.zeros((2, k))
    for i in range(k):
        low, high = metric_cis[i]
        y_err[0, i] = metric_means[i] - low
        y_err[1, i] = high - metric_means[i]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bars = ax.bar(metric_names, metric_means, yerr=y_err, capsize=5, alpha=0.8)
    ax.set_ylabel("Value")
    ax.set_title("Bootstrap Performance Summary (95% CI)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{metric_means[i]:.3f}', ha='center', va='bottom')

    fig.tight_layout()
    _save_and_maybe_show(fig, filename)


# =============================================================================
# Parsing and graph / geometry utilities
# =============================================================================

def parse_hbond_file(hbond_filepath):
    """
    Parse per-residue H-bond counts from *_res_by_res_hbonds.dat.
    Returns a list of floats.
    """
    hbond_counts = []
    try:
        with open(hbond_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        hbond_counts.append(float(parts[1]))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[WARN] H-bond parse error {hbond_filepath}: {e}")
    return hbond_counts


def parse_pdb_file(pdb_filepath):
    """
    Parse CA atoms from a PDB file.
    Returns a list of tuples: (res_idx, x, y, z).
    """
    ca_coordinates = []
    processed_indices = set()
    try:
        with open(pdb_filepath, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        res_idx = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        alt_loc = line[16]  # alternate location indicator

                        # Skip alternate locations other than 'A' or blank
                        if alt_loc not in (' ', 'A'):
                            continue

                        if res_idx not in processed_indices:
                            ca_coordinates.append((res_idx, x, y, z))
                            processed_indices.add(res_idx)
                    except (ValueError, IndexError):
                        continue
        ca_coordinates.sort(key=lambda item: item[0])
    except Exception as e:
        print(f"[WARN] PDB parse error {pdb_filepath}: {e}")
    return ca_coordinates


def compute_graph_features(ca_coordinates, distance_threshold=6.7):
    """
    Build a contact graph from CA coordinates and compute:
      - total edge length
      - average degree
      - average clustering coefficient
      - degree & clustering per residue
      - geometric features per residue in PCA-aligned spherical coordinates:
        radius (r), polar angle (theta), azimuthal angle (phi).
    Returns:
      (total_length, avg_degree, avg_clustering,
       degrees_dict, clustering_dict,
       sph_r_dict, sph_theta_dict, sph_phi_dict)
    """
    if not ca_coordinates:
        return (0, 0.0, 0.0, {}, {}, {}, {}, {})

    try:
        coords_array = np.array([c[1:] for c in ca_coordinates], dtype=float)
        res_nums = [int(c[0]) for c in ca_coordinates]
        n_nodes = len(res_nums)
        if n_nodes == 0:
            return (0, 0.0, 0.0, {}, {}, {}, {}, {})
    except Exception as e:
        print(f"[ERROR] compute_graph_features coords: {e}")
        return (0, 0.0, 0.0, {}, {}, {}, {}, {})

    # Graph construction
    G = nx.Graph()
    G.add_nodes_from(res_nums)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(coords_array[i] - coords_array[j])
            if dist < distance_threshold:
                G.add_edge(res_nums[i], res_nums[j], weight=dist)

    try:
        total_length = float(sum(nx.get_edge_attributes(G, 'weight').values()))
    except Exception:
        total_length = 0.0

    try:
        raw_degrees = dict(G.degree())
        degrees_dict = {res: raw_degrees.get(res, 0) for res in res_nums}
    except Exception as e:
        print(f"[DEBUG] Error computing degrees: {e}")
        degrees_dict = {res: 0 for res in res_nums}

    try:
        raw_clustering = nx.clustering(G, weight='weight')
        clustering_dict = {res: raw_clustering.get(res, 0.0) for res in res_nums}
    except Exception as e:
        print(f"[DEBUG] Error computing clustering: {e}")
        clustering_dict = {res: 0.0 for res in res_nums}

    avg_degree = sum(degrees_dict.values()) / n_nodes if n_nodes > 0 else 0.0
    avg_clustering = sum(clustering_dict.values()) / n_nodes if n_nodes > 0 else 0.0

    # Geometric features: PCA alignment + spherical coordinates
    try:
        # Center coordinates
        coords_centered = coords_array - coords_array.mean(axis=0, keepdims=True)
        # PCA via covariance eigen-decomposition
        cov = np.cov(coords_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        coords_pca = coords_centered @ eigvecs

        x_p = coords_pca[:, 0]
        y_p = coords_pca[:, 1]
        z_p = coords_pca[:, 2]
        r = np.sqrt(x_p**2 + y_p**2 + z_p**2)
        theta = np.zeros_like(r)
        mask = r > 1e-8
        theta[mask] = np.arccos(np.clip(z_p[mask] / r[mask], -1.0, 1.0))
        phi = np.arctan2(y_p, x_p)

        sph_r_dict = {res_nums[i]: float(r[i]) for i in range(n_nodes)}
        sph_theta_dict = {res_nums[i]: float(theta[i]) for i in range(n_nodes)}
        sph_phi_dict = {res_nums[i]: float(phi[i]) for i in range(n_nodes)}
    except Exception as e:
        print(f"[DEBUG] Error computing spherical coords: {e}")
        sph_r_dict = {res: 0.0 for res in res_nums}
        sph_theta_dict = {res: 0.0 for res in res_nums}
        sph_phi_dict = {res: 0.0 for res in res_nums}

    return (total_length, avg_degree, avg_clustering,
            degrees_dict, clustering_dict,
            sph_r_dict, sph_theta_dict, sph_phi_dict)


def parse_gfp_rsa_dat(filepath):
    """
    Parse lf_full_rsa.dat-like file:
      RES <res_name> <chain> <res_num> <abs> <rel> ...
    Returns dict {res_num: rel_sasa}, with 'N/A' mapped to 0.0.
    """
    sasa_dict = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("RES"):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            res_idx = int(parts[3])
                            if parts[5] == 'N/A':
                                rel_sasa = 0.0
                            else:
                                rel_sasa = float(parts[5])
                            sasa_dict[res_idx] = rel_sasa
                        except (ValueError, IndexError):
                            print(f"[DEBUG] Skipping RSA line: {line.strip()}")
                            continue
    except FileNotFoundError:
        print(f"[WARN] RSA file not found: {filepath}")
    except Exception as e:
        print(f"[WARN] Error reading RSA file {filepath}: {e}")
    return sasa_dict


def parse_single_value_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return float(f.readline().strip())
    except Exception as e:
        print(f"[WARN] Could not read single value from {filepath}: {e}")
        return None


def parse_double_value_file(filepath):
    try:
        with open(filepath, 'r') as f:
            parts = f.readline().strip().split()
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
    except Exception as e:
        print(f"[WARN] Could not read 2 values from {filepath}: {e}")
    return None, None


# =============================================================================
# Dataset builder
# =============================================================================

def find_file_by_suffix(directory, suffix):
    """Find the first file in a directory that ends with the given suffix."""
    try:
        for filename in os.listdir(directory):
            if filename.endswith(suffix):
                return os.path.join(directory, filename)
    except FileNotFoundError:
        pass
    return None


def build_dataset(base_dir):
    """
    Build the GFP dataset, including geometric graph features.
    """
    try:
        mutation_folders = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    except FileNotFoundError:
        print(f"[ERROR] Base directory not found: {base_dir}")
        return pd.DataFrame()

    data_records = []
    all_found_residue_indices = set()

    total_folders = len(mutation_folders)
    print(f"[INFO] Found {total_folders} total folders to check...")

    for i, folder_name in enumerate(mutation_folders):
        print(f"[INFO] Processing folder {i+1}/{total_folders}: '{folder_name}'")
        folder_path = os.path.join(base_dir, folder_name)

        match = re.search(r'res(\d+)_([A-Z]{3})', folder_name)
        if not match:
            continue
        res_index_str, mutated_res = match.groups()
        res_index = int(res_index_str)

        path_target = os.path.join(folder_path, "medianBrightness.dat")
        if not os.path.exists(path_target):
            continue

        feature_files_to_check = [
            find_file_by_suffix(folder_path, "_lf_full_hbonds.dat"),
            find_file_by_suffix(folder_path, "_res_by_res_hbonds.dat"),
            find_file_by_suffix(folder_path, "_lf_full_sasa_cro.dat"),
            find_file_by_suffix(folder_path, "_lf_full_sasa_whole_prot.dat"),
            find_file_by_suffix(folder_path, "_lf_full_his.pdb"),
            find_file_by_suffix(folder_path, "_lf_full.pdb"),
            find_file_by_suffix(folder_path, "_lf_full_rsa.dat"),
        ]

        found_files_count = sum(1 for f in feature_files_to_check if f is not None)
        if found_files_count < 5:
            print(f"  -> folder '{folder_name}' has only {found_files_count}/7 feature files, skipping.")
            continue
        else:
            print(f"  -> folder '{folder_name}' has {found_files_count}/7 feature files, processing.")

        try:
            (path_total_hbonds,
             path_res_hbonds,
             path_sasa_cro,
             path_sasa_whole,
             path_pdb_his,
             path_pdb_full,
             path_rsa) = feature_files_to_check

            hb_val1, hb_val2 = parse_double_value_file(path_total_hbonds) if path_total_hbonds else (None, None)
            sasa_cro_val = parse_single_value_file(path_sasa_cro) if path_sasa_cro else None
            sasa_whole_val = parse_single_value_file(path_sasa_whole) if path_sasa_whole else None
            res_hbonds_list = parse_hbond_file(path_res_hbonds) if path_res_hbonds else []

            coords_his = parse_pdb_file(path_pdb_his) if path_pdb_his else []
            (len_his, avg_deg_his, avg_clust_his,
             deg_his, clust_his,
             sph_r_his, sph_theta_his, sph_phi_his) = compute_graph_features(coords_his)

            coords_full = parse_pdb_file(path_pdb_full) if path_pdb_full else []
            (len_full, avg_deg_full, avg_clust_full,
             deg_full, clust_full,
             sph_r_full, sph_theta_full, sph_phi_full) = compute_graph_features(coords_full)

            rsa_data = parse_gfp_rsa_dat(path_rsa) if path_rsa else {}
            target_val = parse_single_value_file(path_target)

            # Update set of all residue indices observed anywhere
            for d in (deg_his, clust_his, deg_full, clust_full, rsa_data,
                      sph_r_his, sph_theta_his, sph_phi_his,
                      sph_r_full, sph_theta_full, sph_phi_full):
                all_found_residue_indices.update(d.keys())

            temp_record = {
                "base_info": {
                    "MutationID": folder_name,
                    "ResidueIndex": res_index,
                    "MutatedResidue": mutated_res,
                    "TotalHBond_Val1": hb_val1,
                    "TotalHBond_Val2": hb_val2,
                    "SASA_Chromophore": sasa_cro_val,
                    "SASA_Whole_Protein": sasa_whole_val,
                    "TotalLength_His": len_his,
                    "AvgDegree_His": avg_deg_his,
                    "AvgClustering_His": avg_clust_his,
                    "TotalLength_Full": len_full,
                    "AvgDegree_Full": avg_deg_full,
                    "AvgClustering_Full": avg_clust_full,
                    "MedianBrightness": target_val,
                },
                "res_hbonds": {i + 1: val for i, val in enumerate(res_hbonds_list)},
                "degrees_his": deg_his,
                "clustering_his": clust_his,
                "degrees_full": deg_full,
                "clustering_full": clust_full,
                "rsa_data": rsa_data,
                "sph_r_his": sph_r_his,
                "sph_theta_his": sph_theta_his,
                "sph_phi_his": sph_phi_his,
                "sph_r_full": sph_r_full,
                "sph_theta_full": sph_theta_full,
                "sph_phi_full": sph_phi_full,
            }
            data_records.append(temp_record)
        except Exception as e:
            print(f"[ERROR] Failed processing folder {folder_name}: {e}")
            continue

    print(f"[INFO] Finished scanning. Kept {len(data_records)} folders that met the data threshold.")
    if not data_records:
        return pd.DataFrame()

    max_residues_found = max(all_found_residue_indices) if all_found_residue_indices else 0
    all_residue_indices = list(range(1, max_residues_found + 1))
    print(f"[INFO] Max residue index found: {max_residues_found}. "
          f"Creating per-residue features for indices 1..{max_residues_found}.")

    final_records = []
    for temp_rec in data_records:
        record = temp_rec['base_info'].copy()
        for res_k in all_residue_indices:
            record[f'ResHBond_{res_k}']        = temp_rec['res_hbonds'].get(res_k, 0.0)
            record[f'Degree_His_{res_k}']      = temp_rec['degrees_his'].get(res_k, 0.0)
            record[f'Clustering_His_{res_k}']  = temp_rec['clustering_his'].get(res_k, 0.0)
            record[f'Degree_Full_{res_k}']     = temp_rec['degrees_full'].get(res_k, 0.0)
            record[f'Clustering_Full_{res_k}'] = temp_rec['clustering_full'].get(res_k, 0.0)
            record[f'RSA_{res_k}']             = temp_rec['rsa_data'].get(res_k, 0.0)

            # New geometric features
            record[f'SphR_His_{res_k}']        = temp_rec['sph_r_his'].get(res_k, 0.0)
            record[f'SphTheta_His_{res_k}']    = temp_rec['sph_theta_his'].get(res_k, 0.0)
            record[f'SphPhi_His_{res_k}']      = temp_rec['sph_phi_his'].get(res_k, 0.0)
            record[f'SphR_Full_{res_k}']       = temp_rec['sph_r_full'].get(res_k, 0.0)
            record[f'SphTheta_Full_{res_k}']   = temp_rec['sph_theta_full'].get(res_k, 0.0)
            record[f'SphPhi_Full_{res_k}']     = temp_rec['sph_phi_full'].get(res_k, 0.0)

        final_records.append(record)

    df = pd.DataFrame(final_records)
    initial_rows = len(df)
    df.dropna(subset=['MedianBrightness'], inplace=True)
    if len(df) < initial_rows:
        print(f"[INFO] Dropped {initial_rows - len(df)} rows with missing MedianBrightness.")

    # Drop H-bond columns that are all zeros
    hbond_cols = [col for col in df.columns if col.startswith("ResHBond_")]
    cols_to_drop = [col for col in hbond_cols if (df[col].fillna(0) == 0).all()]
    if cols_to_drop:
        print(f"[INFO] Dropping {len(cols_to_drop)} all-zero H-bond columns.")
        df.drop(columns=cols_to_drop, inplace=True)

    # Median imputation for remaining numeric NaNs (excluding index & target)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_exclude = ['ResidueIndex', 'MedianBrightness']
    cols_for_imputation = [c for c in numeric_cols if c not in cols_to_exclude]
    if df[cols_for_imputation].isnull().sum().sum() > 0:
        print("[INFO] Applying median imputation to remaining NaN numeric features...")
        imputer = SimpleImputer(strategy='median')
        df[cols_for_imputation] = imputer.fit_transform(df[cols_for_imputation])

    # Remove zero-variance columns
    print("[INFO] Checking for zero-variance columns...")
    cols_to_check = df.columns.drop(['MutationID'])
    no_variance_cols = [col for col in cols_to_check if df[col].nunique() <= 1]
    if no_variance_cols:
        print(f"[INFO] Dropping {len(no_variance_cols)} zero-variance columns.")
        df.drop(columns=no_variance_cols, inplace=True)
    else:
        print("[INFO] No zero-variance columns found.")

    print(f"[INFO] build_dataset finished. Final DataFrame shape: {df.shape}")
    return df


# =============================================================================
# Training, evaluation, and bootstrap logic
# =============================================================================

def save_metric_summary(best_params,
                        train_metrics,
                        test_metrics,
                        boot_summary,
                        out_dir=OUTPUT_DIR):
    """
    Save a CSV + TXT summary of key metrics and best parameters.
    train_metrics / test_metrics: dict with keys R2, Pearson, MSE, MAE
    boot_summary: dict with keys metric -> (mean, ci_low, ci_high)
    """
    _ensure_output_dir()
    lines = []
    lines.append("Best RandomForest parameters:")
    lines.append(str(best_params))
    lines.append("")
    lines.append("Direct train/test metrics:")
    for k in train_metrics:
        lines.append(f"{k}_train = {train_metrics[k]:.6f}")
        lines.append(f"{k}_test  = {test_metrics[k]:.6f}")
    lines.append("")
    lines.append("Bootstrap test metrics (mean [CI_low, CI_high]):")
    for k, (mean_val, ci_low, ci_high) in boot_summary.items():
        lines.append(f"{k}: {mean_val:.6f} [{ci_low:.6f}, {ci_high:.6f}]")

    txt_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    # Also CSV-style summary
    import csv
    csv_path = os.path.join(out_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "train", "test", "boot_mean", "boot_ci_low", "boot_ci_high"])
        for k in train_metrics:
            m = k
            bm, cl, ch = boot_summary.get(m, (float("nan"), float("nan"), float("nan")))
            writer.writerow([m, train_metrics[m], test_metrics[m], bm, cl, ch])


def main(base_dir="GFP_data"):
    """
    Main routine:
      - Build dataset from base_dir
      - Train RandomForest with GridSearchCV on all features
      - Evaluate on train/test
      - Bootstrap on test and on all data
      - Save metrics and plots to OUTPUT_DIR
    """
    overall_start = time.time()
    print(f"[INFO] Starting dataset build from base directory: {base_dir}")
    df = build_dataset(base_dir)
    if df.empty:
        print("[ERROR] build_dataset returned empty DataFrame. Exiting.")
        return

    if 'MedianBrightness' not in df.columns:
        print("[ERROR] Target column 'MedianBrightness' not found. Exiting.")
        return

    # Drop any remaining NaN targets
    df = df.dropna(subset=['MedianBrightness'])

    # One-hot encode mutated residue identity
    df = pd.get_dummies(df, columns=['MutatedResidue'],
                        prefix='MutRes', dummy_na=False)

    # Feature matrix and target vector
    feature_cols = [c for c in df.columns if c not in ['MutationID', 'MedianBrightness']]
    X = df[feature_cols]
    y = df['MedianBrightness']

    if X.isnull().values.any():
        print("[ERROR] NaNs detected in feature matrix X after preprocessing. Exiting.")
        return

    print(f"[INFO] Final feature matrix shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Train shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[INFO] Test shapes:  X_test={X_test.shape}, y_test={y_test.shape}")

    # GridSearchCV for RandomForest
    print("\n[INFO] Starting GridSearchCV for RandomForestRegressor...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [2, 4]
    }
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid_rf,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    t1 = time.time()
    print(f"[INFO] GridSearchCV finished in {(t1 - t0) / 60:.2f} minutes")
    best_model = grid_search.best_estimator_
    print("[INFO] Best parameters:", grid_search.best_params_)

    # Initial evaluation
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    pear_train = pearsonr(y_train, y_pred_train)[0]
    pear_test = pearsonr(y_test, y_pred_test)[0]
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("\n--- Model Performance (Optimized RandomForest) ---")
    print(f"Metric     | Train       | Test")
    print(f"-----------|-------------|-------------")
    print(f"R²         | {r2_train:>11.4f} | {r2_test:>11.4f}")
    print(f"Pearson R  | {pear_train:>11.4f} | {pear_test:>11.4f}")
    print(f"MSE        | {mse_train:>11.4f} | {mse_test:>11.4f}")
    print(f"MAE        | {mae_train:>11.4f} | {mae_test:>11.4f}")

    # Save scatter plots for the test set
    plot_predictions(y_test.values, y_pred_test,
                     filename="test_predictions_scatter.png")
    plot_predictions_fixed(y_test.values, y_pred_test,
                           filename="test_predictions_fixed_0_4.png")

    # Bootstrap on test set
    print("\n[INFO] Bootstrapping on test set (retraining on bootstrapped train)...")
    n_boot = 100
    boot_test_preds = []
    boot_test_metrics = {'MSE': [], 'MAE': [], 'R2': [], 'Pearson': []}

    for i in range(n_boot):
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        model = RandomForestRegressor(random_state=42 + i, n_jobs=-1,
                                      **grid_search.best_params_)
        model.fit(X_boot, y_boot)
        y_pred = model.predict(X_test)
        boot_test_preds.append(y_pred)
        boot_test_metrics['MSE'].append(mean_squared_error(y_test, y_pred))
        boot_test_metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
        boot_test_metrics['R2'].append(r2_score(y_test, y_pred))
        boot_test_metrics['Pearson'].append(pearsonr(y_test, y_pred)[0])

        if (i + 1) % 10 == 0:
            print(f"  - Completed {i+1}/{n_boot} bootstrap iterations for test set")

    boot_test_preds_np = np.array(boot_test_preds)
    mean_preds_test = boot_test_preds_np.mean(axis=0)
    ci_low_test = np.percentile(boot_test_preds_np, 2.5, axis=0)
    ci_high_test = np.percentile(boot_test_preds_np, 97.5, axis=0)

    # Plot bootstrap CI scatter for test set
    plot_predictions_with_confidence(
        y_test.values,
        mean_preds_test,
        ci_low_test,
        ci_high_test,
        title_suffix="Test_Set",
        filename_prefix="bootstrap_predictions"
    )

    # Bootstrap metrics summary + bar plot
    metric_names = ["MSE", "MAE", "R2", "Pearson"]
    metric_means = []
    metric_cis = []
    boot_summary = {}
    for m in metric_names:
        vals = np.array(boot_test_metrics[m])
        mean_val = vals.mean()
        ci_low = np.percentile(vals, 2.5)
        ci_high = np.percentile(vals, 97.5)
        metric_means.append(mean_val)
        metric_cis.append([ci_low, ci_high])
        boot_summary[m] = (mean_val, ci_low, ci_high)
        print(f"[BOOT] {m}: mean={mean_val:.4f}, 95% CI=({ci_low:.4f}, {ci_high:.4f})")

    plot_bootstrapped_metric_bars(metric_means, metric_cis, metric_names,
                                  filename="bootstrap_test_metrics.png")

    # Optional: bootstrap-based predictions on all data
    print("\n[INFO] Bootstrapping on full dataset for predictions (optional)...")
    all_boot_preds = []
    for i in range(n_boot):
        X_boot, y_boot = resample(X_train, y_train, random_state=100 + i)
        model = RandomForestRegressor(random_state=142 + i, n_jobs=-1,
                                      **grid_search.best_params_)
        model.fit(X_boot, y_boot)
        all_boot_preds.append(model.predict(X))

        if (i + 1) % 10 == 0:
            print(f"  - Completed {i+1}/{n_boot} bootstrap iterations for full data")

    all_boot_preds_np = np.array(all_boot_preds)
    mean_preds_all = all_boot_preds_np.mean(axis=0)
    ci_low_all = np.percentile(all_boot_preds_np, 2.5, axis=0)
    ci_high_all = np.percentile(all_boot_preds_np, 97.5, axis=0)

    plot_predictions_with_confidence(
        y.values,
        mean_preds_all,
        ci_low_all,
        ci_high_all,
        title_suffix="All_Data",
        filename_prefix="bootstrap_predictions"
    )

    # Save metrics summary to files
    train_metrics = {
        "R2": r2_train,
        "Pearson": pear_train,
        "MSE": mse_train,
        "MAE": mae_train
    }
    test_metrics = {
        "R2": r2_test,
        "Pearson": pear_test,
        "MSE": mse_test,
        "MAE": mae_test
    }
    save_metric_summary(grid_search.best_params_, train_metrics, test_metrics, boot_summary)

    overall_end = time.time()
    print(f"\n[INFO] main() finished in {overall_end - overall_start:.1f} seconds.")


# =============================================================================
# Script entry point (CLI for HPC / local runs)
# =============================================================================

if __name__ == "__main__":
    print("[MAIN] Starting execution...")
    parser = argparse.ArgumentParser(description="GFP brightness prediction with geometric graph features.")
    parser.add_argument("--base_dir", type=str, default="GFP_data",
                        help="Base directory containing mutation folders.")
    parser.add_argument("--show_plots", action="store_true",
                        help="If set, display plots in addition to saving them.")
    args = parser.parse_args()

    SHOW_PLOTS = args.show_plots
    main(base_dir=args.base_dir)
    print("[MAIN] Done.")
