import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from biogeme import biogeme as bio
from biogeme import database as db
from biogeme.expressions import (
    Beta, Variable, bioDraws, PanelLikelihoodTrajectory, MonteCarlo
)
from biogeme.models import logit

# ===============================
# LOAD & PREPARE DATA
# ===============================
DATA_PATH = r"C:\Users\52812\Desktop\RWTH\2nd_semester_RWTH\Analytics_project\Project_final\dataset\DataSetModel0NA6Alt.csv"
df = (
    pd.read_csv(DATA_PATH, sep=';')
      .apply(pd.to_numeric, errors='coerce')
      .fillna(0)
)
df.rename(columns={'Choice': 'CHOICE', 'ClubNT': 'clubNT'}, inplace=True)

# Lagged choice for inertia
df['prev_choice'] = (
    df.groupby('ID')['CHOICE']
      .shift(1)
      .fillna(0)
      .astype(int)
)

# Players with inertia
df['has_inertia'] = df['ID'].duplicated().astype(int)

# Scale covariates
for col in ['foot', 'moveGK']:
    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

# ===============================
# BIOGEME SETUP
# ===============================
database = db.Database('penalty', df)
database.panel('ID')

CHOICE0   = database.define_variable('CHOICE0', Variable('CHOICE') - 1)
foot      = Variable('foot')
moveGK    = Variable('moveGK')
prev_eq   = Variable('prev_choice') + 1
has_inertia = Variable('has_inertia')

# ===============================
# PARAMETERS
# ===============================
# ASC
ASC1 = Beta('ASC1', 0.0, None, None, 0)
ASC2 = Beta('ASC2', 0.0, None, None, 1)
ASC3 = Beta('ASC3', 0.0, None, None, 0)
ASC4 = Beta('ASC4', 0.0, None, None, 0)
ASC5 = Beta('ASC5', 0.0, None, None, 1)
ASC6 = Beta('ASC6', 0.0, None, None, 0)

# Random Slope: foot (shared across alts)
mu_foot     = Beta('mu_foot', 0.0, None, None, 0)
sigma_foot  = Beta('sigma_foot', 1.0, 0.0, None, 0)
draw_foot   = bioDraws('draw_foot', 'NORMAL')
b_foot      = mu_foot + sigma_foot * draw_foot

# Random Slope: moveGK (shared across alts)
mu_moveGK     = Beta('mu_moveGK', 0.0, None, None, 0)
sigma_moveGK  = Beta('sigma_moveGK', 1.0, 0.0, None, 0)
draw_moveGK   = bioDraws('draw_moveGK', 'NORMAL')
b_moveGK      = mu_moveGK + sigma_moveGK * draw_moveGK

# Random Inertia
mu_inertia     = Beta('mu_inertia', 0.0, None, None, 0)
sigma_inertia  = Beta('sigma_inertia', 1.0, 0.0, None, 0)
draw_inertia   = bioDraws('draw_inertia', 'NORMAL')
b_inertia      = mu_inertia + sigma_inertia * draw_inertia

# Random Intercept
mu_ri     = Beta('mu_ri', 0.0, None, None, 0)
sigma_ri  = Beta('sigma_ri', 1.0, 0.0, None, 0)
draw_ri   = bioDraws('draw_ri', 'NORMAL')
ri        = mu_ri + sigma_ri * draw_ri

# ===============================
# UTILITY FUNCTIONS
# ===============================
V = {
    0: ASC1 + b_foot*foot + b_moveGK*moveGK + b_inertia*(prev_eq == 1)*has_inertia + ri,
    1: ASC2 + b_foot*foot + b_moveGK*moveGK + b_inertia*(prev_eq == 2)*has_inertia + ri,
    2: ASC3 + b_foot*foot + b_moveGK*moveGK + b_inertia*(prev_eq == 3)*has_inertia + ri,
    3: ASC4 + b_foot*foot + b_moveGK*moveGK + b_inertia*(prev_eq == 4)*has_inertia + ri,
    4: ASC5 + b_foot*foot + b_moveGK*moveGK + b_inertia*(prev_eq == 5)*has_inertia + ri,
    5: ASC6 + b_foot*foot + b_moveGK*moveGK + b_inertia*(prev_eq == 6)*has_inertia + ri,
}

avail = {j: 1 for j in V}

# ===============================
# LIKELIHOOD & ESTIMATION
# ===============================
logprob  = logit(V, avail, CHOICE0)
panel_ll = PanelLikelihoodTrajectory(logprob)
simul_ll = MonteCarlo(panel_ll)

biogeme                 = bio.BIOGEME(database, simul_ll)
biogeme.modelName       = 'mixedlogit_random_inertia_moveGK_foot'
biogeme.number_of_draws = 5000

results = biogeme.estimate()
print(results.getEstimatedParameters())

# ===============================
# OPTIONAL: SIMULATE DISTRIBUTIONS
# ===============================
# Replace with your estimated values if needed:
# Example:
# mu_foot_value = results.getBetaValues()['mu_foot']
# sigma_foot_value = results.getBetaValues()['sigma_foot']
# mu_moveGK_value = results.getBetaValues()['mu_moveGK']
# sigma_moveGK_value = results.getBetaValues()['sigma_moveGK']

# Uncomment to run after model estimation:
# N = 10000
# z = np.random.normal(size=N)
# b_foot_sim = mu_foot_value + sigma_foot_value * z
# b_moveGK_sim = mu_moveGK_value + sigma_moveGK_value * z

# plt.figure(figsize=(10,5))
# plt.hist(b_foot_sim, bins=50, alpha=0.6, label='foot', density=True)
# plt.hist(b_moveGK_sim, bins=50, alpha=0.6, label='moveGK', density=True)
# plt.axvline(0, color='black', linestyle='--')
# plt.title('Simulated Distribution of Random Coefficients')
# plt.legend()
# plt.show()

