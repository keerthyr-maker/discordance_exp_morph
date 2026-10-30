
import pandas as pd
import numpy as np
import scipy
import sklearn
from scipy.stats import multivariate_normal 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


def gaussian_log_likelihood(Y, cat, total_rna, spatial_blocks, group_col=None, shrink_k=30):
    """
    Compute the Gaussian log-likelihood of Y given categorical variables and total RNA, controlling for spatial blocks.
    """
    Yv = Y.values.astype(float)
    pred_cov = np.full(Yv.shape, np.nan)  # covariates only (baseline)

    for tr, te in GroupKFold(spatial_blocks.nunique()).split(Yv, groups=spatial_blocks.values):
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cat.iloc[tr])
        cat_encoded = np.c_[enc.fit_transform(cat), np.log1p(total_rna.values)]  # Add log(total_rna) as a feature

        lrY = LinearRegression().fit(cat_encoded[tr], Yv[tr])
        cov_part = lrY.predict(cat_encoded[te])
        pred_cov[te] = cov_part

    idx, cols = Y.index, Y.columns
    pred_cov = pd.DataFrame(pred_cov, index=idx, columns=cols)
    resid = Y - pred_cov
    llh = -0.5 * np.sum(resid ** 2) / shrink_k  # Simplified log-likelihood computation

    return dict(pred=pred_cov, resid=Y - pred_cov,
                log_likelihood=pd.Series(llh, index=cols))


def ridge_regression(X,Y, cat, total_rna,spatial_blocks, alphas=np.logspace(-1, 4, 12)):

    Xv, Yv = X.values.astype(float), Y.values.astype(float)
    pred_full = np.full(Yv.shape, np.nan)      # covariates + X
    pred_cov = np.full(Yv.shape, np.nan)       # covariates only (baseline)

    """
    Perform ridge regression with cross-validation to predict Y from X, controlling for categorical variables and total RNA.
    """
    for tr, te in GroupKFold(spatial_blocks.nunique()).split(Xv, groups=spatial_blocks.values):

    # One-hot encode categorical variables
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cat.iloc[tr])
        cat_encoded = np.c_[enc.fit_transform(cat),np.log1p(total_rna.values)]  # Add log(total_rna) as a feature
        
        lrX = LinearRegression().fit(cat_encoded[tr], Xv[tr])               # remove covariate effects from X and Y
        lrY = LinearRegression().fit(cat_encoded[tr], Yv[tr])
        Xtr, Xte = Xv[tr] - lrX.predict(cat_encoded[tr]), Xv[te] - lrX.predict(cat_encoded[te])
        Ytr = Yv[tr] - lrY.predict(cat_encoded[tr])

        sc = StandardScaler().fit(Xtr)
        rr = RidgeCV(alphas=alphas, alpha_per_target=True).fit(sc.transform(Xtr), Ytr)

        cov_part = lrY.predict(cat_encoded[te])
        pred_cov[te] = cov_part
        pred_full[te] = cov_part + rr.predict(sc.transform(Xte))   # covariate part + X-driven part

    idx, cols = Y.index, Y.columns
    pred_full, pred_cov = (pd.DataFrame(a, index=idx, columns=cols) for a in (pred_full, pred_cov))
    r2_full = r2_score(Y, pred_full, multioutput='raw_values')
    r2_cov = r2_score(Y, pred_cov, multioutput='raw_values')

        
    return dict(pred=pred_full, pred_cov_only=pred_cov, resid=Y - pred_full,
                r2_gain=pd.Series(r2_full - r2_cov, index=cols), r2_full=pd.Series(r2_full, index=cols),
                r2_cov_only=pd.Series(r2_cov, index=cols))

def nll_from_resid(resid):
    """
    Compute the negative log-likelihood from residuals.
    """
    Rv = resid.values - resid.values.mean(axis=0)
    S = np.atleast_2d(np.cov(Rv, rowvar=False)) + 1e-6 * np.eye(Rv.shape[1])
    nll = -np.atleast_1d(multivariate_normal(np.zeros(Rv.shape[1]), S, allow_singular=True).logpdf(Rv))
    return pd.Series(nll, index=resid.index)
