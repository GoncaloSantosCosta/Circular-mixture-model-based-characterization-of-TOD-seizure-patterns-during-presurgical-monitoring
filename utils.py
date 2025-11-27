import numpy as np
import vonMisesMixtures as vonmises
import pandas as pd


def time_to_rad(df):
    """
    Converts ONSET imes into radians [-pi,pi).

    Parameters
    ----------
    df : pd.Series
        Series with the ONSET times in datetime format

    Returns
    -------
    radians : pd.Series
        Series with the ONSET times in radians
    """
    
    # Convert ONSET times to minutes since midnight
    minutes_since_midnight = df.apply(lambda x: x.hour * 60 + x.minute + x.second / 60)

    # Convert times to radians for circular analysis
    radians = (minutes_since_midnight / (24 * 60)) * 2 * np.pi - np.pi  # Convert minutes into radians directly
    
    return radians


def compute_icl_bic(radians, params):
    """
    Computes log-likelihood, Bayesian Information Criterion (BIC),
    and Integrated Completed Likelihood (ICL) for a fitted
    von Mises mixture model.
    
    Parameters
    ----------
    radians : array-like
        Array of observed angular data in radians.
    params : np.ndarray
        (3, K) array from `vonmises.mixture_pdfit`, where:
            params[0, k] = mixture weight πₖ
            params[1, k] = mean direction μₖ
            params[2, k] = concentration parameter κₖ
    
    Returns
    -------
    dict
        Dictionary containing:
            - 'loglik' : float, total log-likelihood
            - 'BIC' : float, Bayesian Information Criterion
            - 'ICL' : float, Integrated Completed Likelihood
            - 'weights' : array of component weights
            - 'mus' : array of mean directions
            - 'kappas' : array of concentration parameters
    """

    weights = np.asarray(params[0, :], dtype=float)
    mus     = np.asarray(params[1, :], dtype=float)
    kappas  = np.asarray(params[2, :], dtype=float)
    
    # Normalize weights in case of tiny numerical drift
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    
    n = len(radians)       # no of samples
    K = weights.size       # no of clusters
    
    # Density of the mixtures (sum of component density * weight)
    comp = np.zeros((n, K), dtype=float)
    for k in range(K):
        component_density = vonmises.density(radians, mu=mus[k], kappa=kappas[k])
        comp[:, k] = weights[k] * component_density
        
    mix = comp.sum(axis=1)
    eps = 1e-300                  # prevents division by 0
    mix = np.maximum(mix, eps)
    
    # Calculate the log likelihood
    loglik = np.sum(np.log(mix))
    
    # Parameter count: (K-1) weights + K means + K kappas = 3K - 1
    p = 3 * K - 1
    
    # BIC calculation
    bic = -2.0 * loglik + p * np.log(n)
    
    # Responsibilities tau for entropy calculation
    tau = comp / mix[:, None]
    
    # Classification entropy
    entropy = -np.sum(tau * np.log(np.maximum(tau, eps)))
    icl = bic + 2.0 * entropy

    return dict(loglik=loglik, BIC=bic, ICL=icl,
                weights=weights, mus=mus, kappas=kappas
                )

        
def pick_majority_lowest_icl(best_fit_runs):
    """
    Selects the model run with the majority number of clusters (K)
    and the lowest ICL among those.
    
    Parameters
    ----------
    best_fit_runs : list of dict
        List of mixture fit results, each containing at least
        the number of clusters 'K' and 'ICL'.
    
    Returns
    -------
    pd.Series
        The entry corresponding to the most common K (majority)
        and the lowest ICL within that subset.
    """
    df = pd.DataFrame(best_fit_runs)
    counts = df["K"].value_counts()
    K_majority = counts.index[counts.values == counts.max()].min()
    subset = df[df["K"] == K_majority]
    return subset.loc[subset["ICL"].idxmin()]


def bic_evaluation(sel, n):
    """
    Evaluates the strength of evidence for clustering using the
    difference in BIC (ΔBIC) between the fitted mixture and a
    uniform (non-clustered) model.
    
    Parameters
    ----------
    sel : dict
        Dictionary containing the BIC of the selected mixture model.
    n : int
        Number of data points used in the fit.
    
    Returns
    -------
    tuple
        (delta_bic, label)
        where:
            - delta_bic : float, BIC difference relative to uniform
            - label : str, qualitative interpretation of evidence
              ('inconclusive', 'weak', 'positive', 'strong', or 'very strong')
    """
    bic_uniform = 2 * n * np.log(2 * np.pi)  # log L0 = -n log(2π)
    delta_bic = bic_uniform - sel["BIC"]
    # print(f"\nΔBIC (uniform − selected mixture) = {delta_bic:.2f}")
    label = ("very strong" if delta_bic >= 10 else
              "strong"     if delta_bic >= 6 else
              "positive"   if delta_bic >= 2 else
              "weak"       if delta_bic > 0 else
              "inconclusive")
    print(f"Evidence for clustering (by ΔBIC): {label}")
    
    return delta_bic, label