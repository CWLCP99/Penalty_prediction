import pandas as pd

from biogeme import biogeme as bio
from biogeme import database as db
from biogeme.expressions import (
    Beta, Variable, bioDraws, PanelLikelihoodTrajectory, MonteCarlo
)
from biogeme.models import logit

# ==================================================
# MIXED LOGIT: ALT-SPECIFIC FOOT & MOVEGK + COMMON ClubNT + GLOBAL INERTIA + RANDOM INTERCEPT
# ON UNBALANCED PANEL
# ==================================================

# 1) LOAD & PREPARE DATA
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

# Scale continuous covariates
for col in ['foot', 'moveGK']:
    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

# Ensure ClubNT is binary (0/1)
# If ClubNT column exists with values 0/1, just rename; otherwise create dummy
# df['clubNT'] = (df['ClubNT'] > 0).astype(int)
df.rename(columns={'ClubNT': 'clubNT'}, inplace=True)

# 2) BUILD DATABASE & DEFINE VARIABLES
# --------------------------------------------------
database = db.Database('penalty', df)
database.panel('ID')  # handles unbalanced panel
CHOICE0   = database.define_variable('CHOICE0', Variable('CHOICE') - 1)

foot      = Variable('foot')
moveGK    = Variable('moveGK')
prev_eq   = Variable('prev_choice') + 1  # equals alt index if repeating

# 3) PARAMETERS
# --------------------------------------------------
# Alt-specific constants (ASC2 & ASC5 normalized to zero)
ASC1 = Beta('ASC1', 0.0, None, None, 0)
ASC2 = Beta('ASC2', 0.0, None, None, 1)
ASC3 = Beta('ASC3', 0.0, None, None, 0)
ASC4 = Beta('ASC4', 0.0, None, None, 0)
ASC5 = Beta('ASC5', 0.0, None, None, 1)
ASC6 = Beta('ASC6', 0.0, None, None, 0)

# Alt-specific slopes for foot (fix 2 & 5)
b_foot1 = Beta('b_foot1', 0.0, None, None, 0)
b_foot2 = Beta('b_foot2', 0.0, None, None, 1)
b_foot3 = Beta('b_foot3', 0.0, None, None, 0)
b_foot4 = Beta('b_foot4', 0.0, None, None, 0)
b_foot5 = Beta('b_foot5', 0.0, None, None, 1)
b_foot6 = Beta('b_foot6', 0.0, None, None, 0)
# Alt-specific slopes for moveGK (fix 2 & 5)
b_move1 = Beta('b_move1', 0.0, None, None, 0)
b_move2 = Beta('b_move2', 0.0, None, None, 1)
b_move3 = Beta('b_move3', 0.0, None, None, 0)
b_move4 = Beta('b_move4', 0.0, None, None, 0)
b_move5 = Beta('b_move5', 0.0, None, None, 1)
b_move6 = Beta('b_move6', 0.0, None, None, 0)


# Global fixed inertia coefficient
global_alpha = Beta('alpha', 0.0, None, None, 0)

# Random intercept per player: ri = mu_ri + sigma_ri * zeta
mu_ri    = Beta('mu_ri',    0.0, None, None, 0)
sigma_ri = Beta('sigma_ri', 1.0,  0.0, None, 0)
zeta     = bioDraws('zeta', 'NORMAL')
ri       = mu_ri + sigma_ri * zeta

# 4) UTILITY SPECIFICATION
# --------------------------------------------------
V = {
    0: ASC1 + b_foot1*foot + b_move1*moveGK + global_alpha*(prev_eq == 1) + ri,
    1: ASC2 + b_foot2*foot + b_move2*moveGK + global_alpha*(prev_eq == 2) + ri,
    2: ASC3 + b_foot3*foot + b_move3*moveGK + global_alpha*(prev_eq == 3) + ri,
    3: ASC4 + b_foot4*foot + b_move4*moveGK + global_alpha*(prev_eq == 4) + ri,
    4: ASC5 + b_foot5*foot + b_move5*moveGK + global_alpha*(prev_eq == 5) + ri,
    5: ASC6 + b_foot6*foot + b_move6*moveGK + global_alpha*(prev_eq == 6) + ri,
}
avail = {j: 1 for j in V}

# 5) LIKELIHOOD & SIMULATION
# --------------------------------------------------
logprob    = logit(V, avail, CHOICE0)
panel_ll   = PanelLikelihoodTrajectory(logprob)
simul_ll   = MonteCarlo(panel_ll)

# 6) ESTIMATION
# --------------------------------------------------
biogeme                     = bio.BIOGEME(database, simul_ll)
biogeme.modelName           = 'mixed_logit_foot_move_clubNT_random_intercept'
biogeme.number_of_draws     = 5000  # increase to 2000+ for final

# 7) ESTIMATE & PRINT
# --------------------------------------------------
results = biogeme.estimate()
print(results.getEstimatedParameters())
