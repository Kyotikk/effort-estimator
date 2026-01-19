"""
Feature Selection Module
========================
Selects top N features using PCA-based energy ranking.
Removes correlated and low-variance features.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def select_features(
    X: pd.DataFrame,
    n_features: int = 50,
    corr_threshold: float = 0.95,
    variance_threshold: float = 1e-8,
) -> dict:
    """
    Select top N features using PCA-based energy ranking.
    
    Args:
        X: Feature matrix DataFrame
        n_features: Number of top features to select
        corr_threshold: Remove features with |correlation| >= threshold
        variance_threshold: Remove features with variance <= threshold
    
    Returns:
        Dictionary with:
            - 'X_selected': Selected feature matrix
            - 'selected_features': List of selected feature names
            - 'pca': Fitted PCA object
            - 'explained_variance': PCA explained variance DataFrame
    """
    # Remove constant features
    var = X.var(axis=0, skipna=True)
    X = X.loc[:, var > variance_threshold]
    logger.info(f"After variance filter: {X.shape[1]} features")
    
    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    
    drop_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= corr_threshold:
                drop_features.add(corr_matrix.columns[j])
    
    X = X.drop(columns=list(drop_features))
    logger.info(f"After correlation filter: {X.shape[1]} features")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for feature ranking
    pca = PCA()
    pca.fit(X_scaled)
    
    # Energy score: contribution of each feature across top PCs
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = min(50, len(cum_var))  # Use top 50 PCs for ranking
    
    loadings = np.abs(pca.components_[:n_pcs, :])
    energy_scores = np.sum(loadings * pca.explained_variance_ratio_[:n_pcs, np.newaxis], axis=0)
    
    # Select top N features
    top_indices = np.argsort(energy_scores)[::-1][:n_features]
    selected_features = [X.columns[i] for i in sorted(top_indices)]
    
    X_selected = X[selected_features].copy()
    
    logger.info(f"Selected {len(selected_features)} top features")
    
    explained_var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(cum_var))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": cum_var,
    })
    
    return {
        "X_selected": X_selected,
        "selected_features": selected_features,
        "pca": pca,
        "explained_variance": explained_var_df,
    }
