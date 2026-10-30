import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

def heldout_cca(M, E, cov_df, total_counts, groups, n_comp=3):
    shapes = dict(M=len(M), E=len(E), cov_df=len(cov_df), total_counts=len(total_counts), groups=len(groups))
    assert len(set(shapes.values())) == 1, f"length mismatch: {shapes}"
    assert total_counts.ndim == 1, f"total_counts has shape {total_counts.shape}"
    """M: morph (DataFrame), E: expression PCs (DataFrame), cov_df: categorical covariates,
    total_counts: Series. All share the same index; groups = spatial block defines the folds."""
    folds, n_unseen = [], []
    for tr, te in GroupKFold(groups.nunique()).split(M, groups=groups):
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cov_df.iloc[tr])
        C = np.c_[enc.transform(cov_df), np.log1p(total_counts.values)]       # categoricals + log total counts

        seen = set(map(tuple, cov_df.iloc[tr].values))
        n_unseen.append(sum(tuple(r) not in seen for r in cov_df.iloc[te].values))

        def resid(A):                                                          # covariate fit uses training cells only
            lr = LinearRegression().fit(C[tr], A[tr])
            return A[tr] - lr.predict(C[tr]), A[te] - lr.predict(C[te])

        M_tr, M_te = resid(M.values)
        E_tr, E_te = resid(E.values)
        sm = StandardScaler().fit(M_tr)
        cca = CCA(n_components=n_comp).fit(sm.transform(M_tr), E_tr)
        u, v = cca.transform(sm.transform(M_te), E_te)
        folds.append([np.corrcoef(u[:, i], v[:, i])[0, 1] for i in range(n_comp)])
    return np.mean(folds, axis=0), np.std(folds, axis=0), n_unseen

def get_null_df(df, morph_cols, strata_cols = ['tm_spatial_block', 'tm_celltype','tm_boundary_annotation'], seed =0):
    """
    Get a null dataframe by shuffling the morph columns within strata defined by strata_cols.
    """
    perm_values = np.random.default_rng(seed)
    null_df = df.copy()
    morph_vals = df[morph_cols].values
    perm = np.arange(len(df))

    for ix in df.groupby(list(strata_cols)).indices.values():
        perm[ix] = perm_values.permutation(ix)
    null_df[morph_cols] = morph_vals[perm]
    return null_df

def perm_null_df(null=np.empty((200,3)),E_1,cat1,total_counts_1,groups1,morph_cols,final_df):
    for i in range(200):
        M_b = get_null_df(final_df, morph_cols, seed=i)[morph_cols]
        null[i] = heldout_cca(M_b, E_1, cat1, total_counts_1, groups1, n_comp=3)[0]
    return null


