import pandas as pd

from biogeme import biogeme as bio
from biogeme import database as db
from biogeme.expressions import (
    Beta, Variable, bioDraws, PanelLikelihoodTrajectory, MonteCarlo
)
from biogeme.models import logit

# ==================================================
# COMPLETE SCRIPT: PANEL LOGIT WITH INERTIA & RANDOM INTERCEPT
# ==================================================

# 1) LOAD, CLEAN & PREPARE
# --------------------------------------------------
DATA_PATH = r"C:\Users\52812\Desktop\RWTH\2nd_semester_RWTH\Analytics_project\Project_final\dataset\DataSetModel0NA6Alt.csv"
df = (
    pd.read_csv(DATA_PATH, sep=';')
      .apply(pd.to_numeric, errors='coerce')
      .fillna(0)
)
df.rename(columns={'Choice': 'CHOICE'}, inplace=True)
# Lagged choice for inertia
df['prev_choice'] = (
    df.groupby('ID')['CHOICE']
      .shift(1)
      .fillna(0)
      .astype(int)
)

# 2) SCALE CONTINUOUS COVARIATES
# --------------------------------------------------
for col in ['foot', 'moveGK']:
    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

# 3) BUILD DATABASE & DEFINE VARIABLES
# --------------------------------------------------
database = db.Database('penalty', df)
# Panel structure
database.panel('ID')
# Zero-based choice
CHOICE0 = database.define_variable('CHOICE0', Variable('CHOICE') - 1)

# Covariates
foot        = Variable('foot')
moveGK      = Variable('moveGK')
prev_choice = Variable('prev_choice')
perc        = {i: Variable(f'perc{i}') for i in range(1,7)}

# 4) PARAMETERS
# --------------------------------------------------
# Alternative-specific constants (ASC1–ASC6, ASC6 normalized)
ASC1 = Beta('ASC1', 0.0, None, None, 0)
ASC2 = Beta('ASC2', 0.0, None, None, 1)
ASC3 = Beta('ASC3', 0.0, None, None, 0)
ASC4 = Beta('ASC4', 0.0, None, None, 0)
ASC5 = Beta('ASC5', 0.0, None, None, 0)
ASC6 = Beta('ASC6', 0.0, None, None, 0)

# Alternative-specific foot slopes (b_foot1–b_foot6), fixing center alternatives 2 & 5
b_foot1 = Beta('b_foot1', 0.5, None, None, 0)
b_foot2 = Beta('b_foot2', 0.0, None, None, 1)  # fixed
b_foot3 = Beta('b_foot3', 0.5, None, None, 0)
b_foot4 = Beta('b_foot4', 0.5, None, None, 0)
b_foot5 = Beta('b_foot5', 0.0, None, None, 0)  # fixed
b_foot6 = Beta('b_foot6', 0.5, None, None, 0)

# Common moveGK slope
g_MOVE = Beta('g_MOVE', 0.5, None, None, 0)

# Alternative-specific perc slopes (b_perc1–b_perc6), fixing center alternatives 2 & 5
b_perc1 = Beta('b_perc1', 0.5, None, None, 0)
b_perc2 = Beta('b_perc2', 0.0, None, None, 1)  # fixed
b_perc3 = Beta('b_perc3', 0.5, None, None, 0)
b_perc4 = Beta('b_perc4', 0.5, None, None, 0)
b_perc5 = Beta('b_perc5', 0.0, None, None, 0)  # fixed
b_perc6 = Beta('b_perc6', 0.5, None, None, 0)

# Inertia term
alpha  = Beta('alpha', 0.0, None, None, 0)
# Random intercept sigma_i
sigma_i = Beta('sigma_i', 0.5, 0, 5, 0)
omega    = bioDraws('omega', 'NORMAL_HALTON2')
mu_i     = sigma_i * omega

# 5) UTILITY SPECIFICATION
# --------------------------------------------------
V = {
    0: ASC1 + alpha*(prev_choice==1) + mu_i + b_foot1*foot + g_MOVE*moveGK + b_perc1*perc[1],
    1: ASC2 + alpha*(prev_choice==2) + mu_i + b_foot2*foot + g_MOVE*moveGK + b_perc2*perc[2],
    2: ASC3 + alpha*(prev_choice==3) + mu_i + b_foot3*foot + g_MOVE*moveGK + b_perc3*perc[3],
    3: ASC4 + alpha*(prev_choice==4) + mu_i + b_foot4*foot + g_MOVE*moveGK + b_perc4*perc[4],
    4: ASC5 + alpha*(prev_choice==5) + mu_i + b_foot5*foot + g_MOVE*moveGK + b_perc5*perc[5],
    5: ASC6 + alpha*(prev_choice==6) + mu_i + b_foot6*foot + g_MOVE*moveGK + b_perc6*perc[6],
}
avail = {j:1 for j in V}

# 6) LOGIT & PANEL LIKELIHOOD
# --------------------------------------------------
logprob    = logit(V, avail, CHOICE0)
panel_ll   = PanelLikelihoodTrajectory(logprob)
simul_ll   = MonteCarlo(panel_ll)

# 7) ESTIMATION
# --------------------------------------------------
biogeme = bio.BIOGEME(database, simul_ll)
biogeme.modelName       = 'panel_logit_alt_specific'
biogeme.number_of_draws = 2000

results = biogeme.estimate()
print(results.get_estimated_parameters())


