
# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

###################################
# Required to avoid type3 fonts that break ICML submission pdf
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
###################################

###################################
# Storing colors here
blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
brown = "#a65628"
pink = "#f781bf"
grey = "#999999"
# green = "#4daf4a"
# red = "#e41a1c"
# yellow = "#ffff33"
####################################

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
font = {'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

# %%
df1 = pd.read_csv('./out/mlrepo12.csv')

# %%
metric = 'auc'
tran_param = "{'space': 'prop'}"
x_var = 'n'
if x_var == 'n':
    xlab = 'Training set size'
elif x_var == 'p':
    xlab = 'Dataset dimension'

head_params = [
    "{'model': 'rf'}",
    "{'model': 'xgb'}",
    "{'model': 'maml'}",
    "{'model': 'deepcoda'}",
    "{'model': 'metann'}",
]

y_idx = [
    "{}",
    "{'mult': True, 'conv': 'rand', 'comb': 'rand', 'space': 'clr', 'weight': 0.2, 'factor': 10}",
]
y_idx = [
    "{}",
    "{'conv': 'rand', 'space': 'clr', 'weight': 0.5, 'factor': 10}",
]
y_idx = [
    "{}",
    "{'comb': 'rand', 'space': 'prop', 'weight': 0.5, 'factor': 10}",
]

# Adjust where we blew up n by 10x
df1['n'][df1['aug_params'] == y_idx[1]] = df1['n'][df1['aug_params'] == y_idx[1]] / 10

# Remove the trivial datasets
df1 = df1[np.logical_not(np.isin(df1['data_idx'], [2, 4]))]

# average over seeds
means = df1.pivot_table(values=['auc', 'bacc'], index=['data_idx', 'n', 'p', 'aug_params', 'tran_params', 'dr_params', 'head_params'], aggfunc='mean')
means = means.reset_index()
means = means[means['tran_params'] == tran_param]
print(means[means['head_params'] == "{'model': 'maml', 'aug': 'aitch'}"])


means['aug_params'][means['head_params'] == "{'model': 'maml', 'aug': 'aitch'}"] = y_idx[1]
means['head_params'][means['head_params'] == "{'model': 'maml', 'aug': 'aitch'}"] = "{'model': 'maml'}"

print(means)

fig, axs = plt.subplots(1, 5, figsize=(13, 3))
axs = axs.flatten()
plot_idx = 0
for head_param in head_params[0:5]:
    tmp_df = means[means['head_params'] == head_param]
    tmp_df = tmp_df[tmp_df['aug_params'].isin(y_idx)]
    tmp_df2 = tmp_df.pivot_table(values=metric, index=['data_idx', 'n', 'p'], columns='aug_params', aggfunc='mean')
    tmp_df2.reset_index(inplace=True)
    y = tmp_df2[y_idx[1]] - tmp_df2[y_idx[0]]
    x = tmp_df2[x_var]
    axs[plot_idx].scatter(x, y, label=head_param)
    axs[plot_idx].axhline(0, color='grey', linestyle=':')
    axs[plot_idx].set_ylim(-0.05, 0.11)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axs[plot_idx].plot(np.sort(x),p(np.sort(x)),"-")
    plot_idx += 1
    
axs[0].set_title('RF')
axs[1].set_title('XGB')
axs[2].set_title('mAML')
axs[3].set_title('DeepCoDa')
axs[4].set_title('MetaNN')

# deepcoda has an outlier
axs[3].set_ylim(-0.05, 0.25)

axs[0].set_ylabel('AUC gain from augmentation')
axs[0].set_xlabel(xlab)
axs[1].set_xlabel(xlab)
axs[2].set_xlabel(xlab)
axs[3].set_xlabel(xlab)
axs[4].set_xlabel(xlab)

plt.tight_layout()
plt.savefig('./out/ablation.pdf')
plt.show()
