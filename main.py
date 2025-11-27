import vonMisesMixtures as vonmises
import numpy as np
import pandas as pd
import utils

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Open DataFrame
full_df = pd.read_excel(path)
df0 = full_df.copy()

# Filter out patients with less than 3 seizures
df0 = df0.groupby("PAT_ID").filter(lambda x: len(x) > 2)

# Get all patients
patients = full_df["PAT_ID"].unique()

full_results = []
for pat in patients:

    print(f"\n---- Patient {pat}: ----")
    
    df = df0[df0['PAT_ID'] == pat]['ONSET']
    df = pd.to_datetime(df)
    
    # Convert time to radians
    radians = utils.time_to_rad(df)
    
    # Re-fit 100 times the vMM
    best_fit = []
    for _ in range(100):
        rows = []
        fits = {}
        for K in range(1,5):  # K = 1..4
            try:
                # Fit von Mises mixture model
                params = vonmises.mixture_pdfit(radians, n = K)
                # Compute ICL for comparison
                icl_res = utils.compute_icl_bic(radians, params)
                icl = icl_res["ICL"]
                rows.append({
                    "K": K,
                    "BIC": icl_res["BIC"],
                    "ICL": icl_res["ICL"]
                })
                fits[K] = icl_res
            except Exception as e:
                rows.append({"K": K, "BIC": np.nan, "ICL": np.nan})
                fits[K] = None
                print(f"[warn] mixture fit failed for K={K}: {e}")
        
        fit_result_df = pd.DataFrame(rows).set_index("K").sort_index()
        
        # Select K by minimum ICL
        valid = fit_result_df["ICL"].dropna()
        if not valid.empty:
            K_icl = valid.idxmin()
        else:
            raise RuntimeError("All mixture fits failed; cannot select K by ICL.")
        
        sel_temp = fits[K_icl]
        sel_temp["K"] = K_icl
        best_fit.append(sel_temp)
        del rows, fits, params, icl_res, icl, fit_result_df, valid, K_icl, sel_temp
    
    # K selection
    best_fit_df = pd.DataFrame(best_fit)
    best_fit_df = best_fit_df.sort_values("K").reset_index(drop=True)
    selected_fit = utils.pick_majority_lowest_icl(best_fit_df)
    K_final = selected_fit["K"]
    print(f"Selected K by ICL: {K_final}")
    
    # Turn cluster centers to time -of-day
    mus_hours = ((selected_fit["mus"] + np.pi) / (2 * np.pi)) * 24.0
    mus_hours = np.mod(mus_hours, 24.0)
    
    # ΔBIC vs uniform (for Bayes-factor style evidence without p-values)
    delta_bic, label = utils.bic_evaluation(selected_fit, len(radians))
    
    pat_results = [pat, len(radians), K_final, list(selected_fit["mus"]), list(mus_hours), list(selected_fit["kappas"]),
                    list(selected_fit["weights"]), float(selected_fit["BIC"]), float(delta_bic), label]
    
    full_results.append(pat_results)
    
    del selected_fit, best_fit, best_fit_df, K_final, mus_hours, delta_bic, label, pat_results, df, radians
    
full_results_df = pd.DataFrame(full_results)
full_results_df.columns = ["Patient", "No. of seizures", "K", "miu rad", "miu hour", "k", "pi", "BIC", "dBIC", "Label"]
    