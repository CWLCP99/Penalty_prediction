# Biogeme modules imports
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models, expressions as ex
from biogeme.expressions import Variable

#Matplotlib and pandas
import matplotlib.pyplot as plt
import pandas as pd

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

# 3) ASCs: one per zone; fix 2 & 5 to zero for identification
ASC1 = ex.Beta('ASC1', 0, None, None, 0)
ASC2 = ex.Beta('ASC2', 0, None, None, 1)   # fixed
ASC3 = ex.Beta('ASC3', 0, None, None, 0)
ASC4 = ex.Beta('ASC4', 0, None, None, 0)
ASC5 = ex.Beta('ASC5', 0, None, None, 1)   # fixed
ASC6 = ex.Beta('ASC6', 0, None, None, 0)

# 4) Alternative‐specific foot coefficients
B_foot1 = ex.Beta('B_foot1', 0, None, None, 0)
B_foot2 = ex.Beta('B_foot2', 0, None, None, 0)
B_foot3 = ex.Beta('B_foot3', 0, None, None, 0)
B_foot4 = ex.Beta('B_foot4', 0, None, None, 0)
B_foot5 = ex.Beta('B_foot5', 0, None, None, 0)
B_foot6 = ex.Beta('B_foot6', 0, None, None, 0)

# 5) Alternative‐specific age coefficients
B_age1  = ex.Beta('B_age1',  0, None, None, 0)
B_age2  = ex.Beta('B_age2',  0, None, None, 0)
B_age3  = ex.Beta('B_age3',  0, None, None, 0)
B_age4  = ex.Beta('B_age4',  0, None, None, 0)
B_age5  = ex.Beta('B_age5',  0, None, None, 0)
B_age6  = ex.Beta('B_age6',  0, None, None, 0)

# 6) Utility functions: one line per alternative
V = {
    1: ASC1 + B_foot1*foot_R + B_age1*age,
    2: ASC2 + B_foot2*foot_R + B_age2*age,
    3: ASC3 + B_foot3*foot_R + B_age3*age,
    4: ASC4 + B_foot4*foot_R + B_age4*age,
    5: ASC5 + B_foot5*foot_R + B_age5*age,
    6: ASC6 + B_foot6*foot_R + B_age6*age,
}

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

print("\n=== All Betas (including fixed ones) ===")
# Use the new API
all_betas = results.get_beta_values()
for name in [
    'ASC1','ASC2','ASC3','ASC4','ASC5','ASC6',
    'B_age1','B_age2','B_age3','B_age4','B_age5','B_age6',
    'B_foot1','B_foot2','B_foot3','B_foot4','B_foot5','B_foot6'
]:
    val = all_betas.get(name, 0.0)
    print(f"{name}: {val:.6f}")
