# Biogeme modules imports
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models, expressions as ex
from biogeme.expressions import Variable

#Matplotlib and pandas
import matplotlib.pyplot as plt
import pandas as pd

# Load your trimmed model‐input CSV (must include 'foot_R','age','alt','Choice')
df = pd.read_csv('dataset/penalty_long_format.csv')
df = df.dropna(subset=['foot_R','age','alt','Choice'])
database = db.Database('penalties', df)

# Define parameters
#    ASCs: one per zone; fix the two centre zones (ASC2 & ASC5) to zero via status=1
ASC1 = ex.Beta('ASC1', 0, None, None, 0)  # Top-Left
ASC2 = ex.Beta('ASC2', 0, None, None, 1)  # Top-Center (fixed to 0)
ASC3 = ex.Beta('ASC3', 0, None, None, 0)  # Top-Right
ASC4 = ex.Beta('ASC4', 0, None, None, 0)  # Bottom-Left
ASC5 = ex.Beta('ASC5', 0, None, None, 1)  # Bottom-Center (fixed to 0)
ASC6 = ex.Beta('ASC6', 0, None, None, 0)  # Bottom-Right

B_foot = ex.Beta('B_foot', 0, None, None, 0)
B_age  = ex.Beta('B_age',  0, None, None, 0)

# Variables (Biogeme symbols)
foot_R    = Variable('foot_R')
age       = Variable('age')
choiceVar = Variable('Choice')

# Utilities: one ASC per alt + common covariates
V = {
    1: ASC1 + B_foot*foot_R + B_age*age,
    2: ASC2 + B_foot*foot_R + B_age*age,
    3: ASC3 + B_foot*foot_R + B_age*age,
    4: ASC4 + B_foot*foot_R + B_age*age,
    5: ASC5 + B_foot*foot_R + B_age*age,
    6: ASC6 + B_foot*foot_R + B_age*age,
}

# Availability: all six alternatives available each choice occasion
AV = {i: 1 for i in V.keys()}

# Logit probability
logprob = models.loglogit(V, AV, choiceVar)

# Estimate
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = 'asc_and_covariates'
results = biogeme.estimate()

# Print results (new API)
print(results.get_estimated_parameters())

###### to save results in different formats #############

# 1) Build est_df from whichever API is available
try:
    # Newer DataFrame‐returning API, index = parameter names
    est_df = results.getEstimatedParameters().copy()
    est_df.index.name = 'Name'
    est_df = est_df.reset_index()
except Exception:
    # Fallback to list‐of‐dicts API
    est_list = results.get_estimated_parameters()
    est_df = pd.DataFrame(est_list)

# 2) Pull all Betas (including fixed) and append any missing
all_betas = results.getBetaValues()
for name, val in all_betas.items():
    if name not in est_df['Name'].values:
        est_df = est_df.append({
            'Name': name,
            'Value': val,
            'Rob. Std err': 0.0,
            'Rob. t-test': float('nan'),
            'Rob. p-value': float('nan')
        }, ignore_index=True)

# 3) Mark status
est_df['Status'] = est_df['Rob. Std err'].eq(0.0).map({True: 'Fixed', False: 'Estimated'})

# 4) Save to Excel
with pd.ExcelWriter('penalty_model_results.xlsx') as writer:
    est_df.to_excel(writer, sheet_name='Model Results', index=False)

# 5) Save to PNG
fig, ax = plt.subplots(figsize=(10, len(est_df)*0.5 + 1))
ax.axis('off')
tbl = ax.table(
    cellText=est_df.values,
    colLabels=est_df.columns,
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.savefig('penalty_model_results.png', bbox_inches='tight')

print("Wrote penalty_model_results.xlsx and penalty_model_results.png")