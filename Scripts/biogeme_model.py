# Biogeme modules imports
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models, expressions as ex
from biogeme.expressions import Variable

#Matplotlib and pandas
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# biogeme_model.py

# 1) Load your long‐format model input
df = pd.read_csv('dataset/penalty_long_format.csv')
df = df.dropna(subset=['foot_R','age','alt','Choice'])
database = db.Database('penalties', df)

# 2) Variables (Biogeme symbols)
foot_R    = Variable('foot_R')
age       = Variable('age')
alt       = Variable('alt')
choiceVar = Variable('Choice')

# New flags
fav_flag      = Variable('fav_flag')
greatGK_flag  = Variable('greatGK_flag')
ingame_flag   = Variable('ingame_flag')
loc_home      = Variable('loc_home')
loc_away      = Variable('loc_away')
last_dir      = Variable('last_dir')

# 3) ASCs: one per zone; fix 2 & 5 to zero for identification (CENTER)
ASC1 = ex.Beta('ASC1', 0, None, None, 0)
ASC2 = ex.Beta('ASC2', 0, None, None, 1)   # fixed
ASC3 = ex.Beta('ASC3', 0, None, None, 0)
ASC4 = ex.Beta('ASC4', 0, None, None, 0)
ASC5 = ex.Beta('ASC5', 0, None, None, 1)   # fixed
ASC6 = ex.Beta('ASC6', 0, None, None, 0)

# 4) Alt‐specific foot coefficients
B_foot = {
    j: ex.Beta(f'B_foot{j}', 0, None, None, 0)
    for j in range(1,7)
}

# 5) Alt‐specific age coefficients
B_age = {
    j: ex.Beta(f'B_age{j}',  0, None, None, 0)
    for j in range(1,7)
}

# 6) Alt‐specific fav_flag coefficients
B_fav = {
    j: ex.Beta(f'B_fav{j}',  0, None, None, 0)
    for j in range(1,7)
}

# 7) Alt‐specific greatGK_flag coefficients
B_gk = {
    j: ex.Beta(f'B_gk{j}',   0, None, None, 0)
    for j in range(1,7)
}

# 8) Alt‐specific ingame_flag coefficients
B_ing = {
    j: ex.Beta(f'B_ing{j}',  0, None, None, 0)
    for j in range(1,7)
}

# 9) Alt‐specific loc_home coefficients
B_home = {
    j: ex.Beta(f'B_home{j}', 0, None, None, 0)
    for j in range(1,7)
}

# 10) Alt‐specific loc_away coefficients
B_away = {
    j: ex.Beta(f'B_away{j}', 0, None, None, 0)
    for j in range(1,7)
}

# 11) Alt‐specific stickiness coefficients
B_last = {
    j: ex.Beta(f'B_last{j}', 0, None, None, 0)
    for j in range(1,7)
}

# 12) Utility functions
V = {}
for j, ASCj in zip(range(1,7), [ASC1,ASC2,ASC3,ASC4,ASC5,ASC6]):
    stick = (last_dir == j)
    V[j] = (
        ASCj
      + B_foot[j] * foot_R
      + B_age[j]  * age
      + B_fav[j]  * fav_flag
      + B_gk[j]   * greatGK_flag
      + B_ing[j]  * ingame_flag
      + B_home[j] * loc_home
      + B_away[j] * loc_away
      + B_last[j] * stick
    )
    
########### ASC ONLY SPECIFICATION ###########

# ASCs (2 & 5 fixed)
V_asc = {
    1: ASC1,
    2: ASC2,
    3: ASC3,
    4: ASC4,
    5: ASC5,
    6: ASC6,
}
AV_asc = {i: 1 for i in V_asc}
logprob_asc = models.loglogit(V_asc, AV_asc, choiceVar)
biogeme_asc = bio.BIOGEME(database, logprob_asc)
biogeme_asc.modelName = 'asc_only'
results_asc = biogeme_asc.estimate()

####### FULL SPECIFIC MODEL ###########

# 7) Availability: all six always available
AV = {i: 1 for i in V.keys()}

# 8) Define and estimate the logit model
logprob  = models.loglogit(V, AV, choiceVar)
biogeme  = bio.BIOGEME(database, logprob)
biogeme.modelName = 'asc_foot_age_alt_specific'
results  = biogeme.estimate()

# 9) Print out the full set of Betas (including fixed)
print("=== Estimated parameters ===")
print(results.get_estimated_parameters())

#print("\n=== All Betas (including fixed ones) ===")
all_betas = results.get_beta_values()
for name in [
    # ASCs
    'ASC1','ASC2','ASC3','ASC4','ASC5','ASC6',
    # Age by alternative
    'B_age1','B_age2','B_age3','B_age4','B_age5','B_age6',
    # Foot by alternative
    'B_foot1','B_foot2','B_foot3','B_foot4','B_foot5','B_foot6',
    # Favored‐team by alternative
    #'B_fav1','B_fav2','B_fav3','B_fav4','B_fav5','B_fav6',
    # Great‐GK by alternative
    #'B_gk1','B_gk2','B_gk3','B_gk4','B_gk5','B_gk6',
    # Ingame vs. shootout by alternative
    #'B_ing1','B_ing2','B_ing3','B_ing4','B_ing5','B_ing6',
    # Home venue by alternative
    #'B_home1','B_home2','B_home3','B_home4','B_home5','B_home6',
    # Away venue by alternative
    #'B_away1','B_away2','B_away3','B_away4','B_away5','B_away6',
    # Stickiness (last_dir) by alternative
    #'B_last1','B_last2','B_last3','B_last4','B_last5','B_last6',
]:
    val = all_betas.get(name, 0.0)
    #print(f"{name}: {val:.6f}")
    
###### MODEL SUMMARIES ######
# Print model summaries
print("\n=== Model summaries ===")
stats_asc = results_asc.get_general_statistics()
stats     = results.get_general_statistics()

ll_asc = stats_asc['Final log likelihood'].value
ll_full= stats   ['Final log likelihood'].value

aic_asc = stats_asc['Akaike Information Criterion'].value
bic_asc = stats_asc['Bayesian Information Criterion'].value

aic_full= stats   ['Akaike Information Criterion'].value
bic_full= stats   ['Bayesian Information Criterion'].value

print(f"ASC-only final log likelihood:    {ll_asc:.2f}")
print(f"Full model final log likelihood:   {ll_full:.2f}\n")

print(f"ASC-only    AIC: {aic_asc:.2f},  BIC: {bic_asc:.2f}")
print(f"Full model  AIC: {aic_full:.2f},  BIC: {bic_full:.2f}")

if ll_full > ll_asc: # Higher log-likelihood is better
    print("Full model has a better fit (log-likelihood) than ASC-only model. ")
    print(ll_full - ll_asc, "improvement in log likelihood.")
else:
    print("ASC-only model has a better fit than full model.")
    
if aic_full < aic_asc: # Lower AIC is better
    print("Full model has a lower AIC than ASC-only model.")
    print(aic_full - aic_asc, "improvement in AIC.")
else: 
    print("ASC-only model has a lower AIC than full model.")
    
if bic_full < bic_asc: # Lower BIC is better
    print("Full model has a lower BIC than ASC-only model.")
    print(bic_full - bic_asc, "improvement in BIC.")
else:
    print("ASC-only model has a lower BIC than full model.")
    
    
####### LOG MODEL SUMMARY TO EXCEL FILE #######

def log_model_summary(results, model_label, excel_path='results/model_summaries.xlsx'):
    # 1) Pull general stats
    stats = results.get_general_statistics()
    ll    = stats['Final log likelihood'].value
    aic   = stats['Akaike Information Criterion'].value
    bic   = stats['Bayesian Information Criterion'].value
    k     = stats['Number of estimated parameters'].value
    n     = stats['Sample size'].value

    # 2) Build a single‐row DataFrame
    df_row = pd.DataFrame([{
        'Model'              : model_label,
        'Params (K)'         : k,
        'Sample size (N)'    : n,
        'LogLik'             : ll,
        'AIC'                : aic,
        'BIC'                : bic,
    }])

    # 3) If file exists, append; otherwise create new
    file = Path(excel_path)
    if file.exists():
        # append without writing the header
        with pd.ExcelWriter(file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # read existing to find out how many rows it has
            existing = pd.read_excel(file, sheet_name='Summary')
            startrow = len(existing) + 1
            df_row.to_excel(writer, sheet_name='Summary', index=False, header=False, startrow=startrow)
    else:
        # first run: write with header
        with pd.ExcelWriter(file, engine='openpyxl') as writer:
            df_row.to_excel(writer, sheet_name='Summary', index=False)

# --- Example usage after each model run ---
log_model_summary(results_asc,  'ASC-only',          'results/model_summaries.xlsx')
log_model_summary(results,    'Full model_6_covariates',        'results/model_summaries.xlsx')

# Add this temporarily to your script right after you get stats_asc:
#print(stats_asc.keys())


