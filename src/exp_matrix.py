import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(df):
    df_filtered = df[
    (df['assignment_status'] == 'assigned')
    & (df['within_retained_cell_status'] != 'source_cell_not_retained')
    & (df['source_cell_status'] == 'retained_target_cell')
]
    return df_filtered

def create_expression_matrix(df_filtered):
    expression_matrix = df_filtered.groupby(['mask_label','geneID']).size().unstack(fill_value=0)
    return expression_matrix

def expression_pca(expression_matrix, n_components=20):
    counts = expression_matrix.select_dtypes('number')            # cells x genes, raw counts
    total_counts = counts.sum(axis=1)

    keep = total_counts >= 20                                  # drop cells with too few molecules
    counts, total_counts = counts[keep], total_counts[keep]
    print(f"dropped {(~keep).sum()} cells, kept {keep.sum()}")

    norm = counts.div(total_counts, axis=0) * 1e4
    logged = np.log1p(norm)

    scaled = StandardScaler().fit_transform(logged)
    scaled = np.clip(scaled, -10, 10)                          # limits the pull of rare, noisy genes

    pca = PCA(n_components=20, svd_solver='arpack', random_state=0)
    expression_pcs = pd.DataFrame(pca.fit_transform(scaled), index=counts.index,
                    columns=[f"PC{i+1}" for i in range(20)])

    return expression_pcs
