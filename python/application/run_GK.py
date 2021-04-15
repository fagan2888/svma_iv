import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]
})
import pandas as pd
from python.functions.SVMAIV import *
import os

"""
1- Load data ----------------------------------------------------------------------------------------------------------+
"""
data = {}

# settings
data['file'] = './gk2015.csv'  # Data file
data['sample'] = [pd.Timestamp('1990-1-1'), pd.Timestamp('2012-06-01')]  # Sample end poitns
data['endo_vars'] = ['ff', 'dlogip', 'dlogcpi', 'ebp']  # Endogenous y variables
data['iv_var'] = 'ff4_tc'  # External IV

# Load
data['table'] = pd.read_csv(data['file'])
data['sample_bool'] = (pd.to_datetime(data['table'].date) >= data['sample'][0]) & (pd.to_datetime(data['table'].date) <= data['sample'][1])

# Select variables
data['Y'] = data['table'].loc[data['sample_bool'], data['endo_vars']]
data['Z'] = data['table'].loc[data['sample_bool'], data['iv_var']]


"""
2- SVMA-IV inference ----------------------------------------------------------------------------------------------------------+
"""
print('*** SVMA-IV analysis ***')

settings = {'ic' : 'aic' ,      # Information criterion
            'n_boot': 500,     # Number of bootstrap samples
            'signif': 0.1,      # Significance level
            'horiz': np.arange(0,20),
            'p': 6}

data_y = data['Y'].to_numpy()
data_z = data['Z'].to_numpy()
np.random.seed(2018)

bounds, id_recov, inv_test, settings_struct, _ = SVMAIV_estim(data_y, data_z, settings)

## Display pre-test for invertibility
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(inv_test.pval.all, columns=['p-value'], index=['all equations jointly'])
df.index.name = 'Invertibility pre-test: '
print(df)
print('\n'+'-'*60+'\n')

df = pd.DataFrame(inv_test.pval.eqns.T, columns=['p-value'])
df.index.name = 'Invertibility pre-test: each equation separately'
print(df)
print('\n'+'-'*60+'\n')

## Display bounds on alpha and degree of invertibility/recoverability
# Scale parameter
df = pd.DataFrame([(bounds['estim']['lower']['alpha'], bounds['estim']['upper']['alpha'])], columns=['']*2,
                  index=['Bound estimates: alpha'])
print(df)
print('\n'+'-'*60+'\n')
df = pd.DataFrame([(bounds['ci']['lower']['alpha'], bounds['ci']['upper']['alpha'])], columns=['']*2,
                  index=['Confidence interval: alpha'])
print(df)
print('\n'+'-'*60+'\n')

# Degree of invertibility
print('Bound estimates: degree of invertibility')
print(bounds['estim']['lower']['R2_inv'], bounds['estim']['upper']['R2_inv'])
print('\n'+'-'*60+'\n')
print('Confidence interval: degree of invertibility')
print(bounds['ci']['lower']['R2_inv'], bounds['ci']['upper']['R2_inv'])
print('\n'+'-'*60+'\n')

# Degree of recoverability
print('Bound estimates: degree of recoverability')
print(bounds['estim']['lower']['R2_recov'], bounds['estim']['upper']['R2_recov'])
print('\n'+'-'*60+'\n')
print('Confidence interval: degree of recoverability')
print(bounds['ci']['lower']['R2_recov'], bounds['ci']['upper']['R2_recov'])

"""
3- Plotting FVR bounds----------------------------------------------------------------------------------------------------------+
"""
if not os.path.isdir('./Figures'):
    os.mkdir('./Figures/')
# Settings
plots = {}
plots['xticks'] = np.arange(0, 21, 4)# X axis ticks for FVR plot
plots['yticks'] = np.arange(0, 1.1, 0.1)
plots['titles'] = ['FVR of Federal Funds Rate', 'FVR of Industrial Production Growth', 'FVR of CPI Growth', 'FVR of Excess Bond Premium']
plots['xlabel'] = 'Horizon (Months)' # X axis label for FVR plot
plots['ylabel'] = '' # Y axis label for FVR plot

for i in range(np.size(data_y, 1)): # For each macro variable...
    # Plot bound estimates and CI for identified set
    fig, axes = plt.subplots(1,1,figsize=(6,5))
    axes.plot(np.arange(0, 24), bounds['ci']['upper']['FVR'][:, i], color='grey', lw=0.7, dashes=[6, 3])
    axes.plot(np.arange(0, 24), bounds['ci']['lower']['FVR'][:, i], color='grey', lw=0.7, dashes=[6, 3],
              label=str(100*(1-settings[
        'signif'])) +
    '\% conf. interval of identif. set')
    axes.plot(np.arange(0,24), bounds['estim']['lower']['FVR'][:,i], 'k', label='Estimate of identif. set')
    axes.plot(np.arange(0, 24), bounds['estim']['upper']['FVR'][:, i], 'k')
    axes.set_ylim(0, 1)
    axes.set_title(plots['titles'][i])
    axes.set_xlabel(plots['xlabel'])
    axes.set_xticks(plots['xticks'])
    axes.set_yticks(plots['yticks'])
    axes.legend()
    plt.grid(alpha=0.3)
    fig.savefig('./Figures/fvr_'+plots['titles'][i]+'.pdf', dpi=800)
