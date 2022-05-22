
# %%
import numpy as np
import pandas as pd


# %%
metric = 'auc' # ece, bacc
tran_param = "{'space': 'prop'}"
aggfunc = 'mean' # std

y_idx = [
    "{}",
    "{'conv': 'rand', 'space': 'clr', 'weight': 0.5, 'factor': 10}",
]
y_idx = [
    "{}",
    "{'subc': True, 'space': '', 'weight': 0.5, 'factor': 10}",
]
y_idx = [
    "{}",
    "{'comb': 'rand', 'space': 'prop', 'weight': 0.5, 'factor': 10}",
]

# %%
df1 = pd.read_csv('./out/mlrepo12.csv')

# average over seeds
means = df1.pivot_table(
    values=[metric], 
    index=['data_idx', 'aug_params', 'tran_params', 'dr_params', 'head_params'], 
    aggfunc=aggfunc
)
if aggfunc == 'std':
    means = means / np.sqrt(20)

means = means.reset_index()
means = means[means['tran_params'] == tran_param]

head_params = [
    "{'model': 'rf'}",
    "{'model': 'xgb'}",
    "{'model': 'maml'}",
    "{'model': 'deepcoda'}",
    "{'model': 'metann'}",
]

means['aug_params'][means['head_params'] == "{'model': 'maml', 'aug': 'aitch'}"] = "{'conv': 'rand', 'space': 'clr', 'weight': 0.5, 'factor': 10}"
means['head_params'][means['head_params'] == "{'model': 'maml', 'aug': 'aitch'}"] = "{'model': 'maml'}"

means['aug_params'][means['head_params'] == "{'model': 'maml', 'aug': 'subc'}"] = "{'subc': True, 'space': '', 'weight': 0.5, 'factor': 10}"
means['head_params'][means['head_params'] == "{'model': 'maml', 'aug': 'subc'}"] = "{'model': 'maml'}"

means['aug_params'][means['head_params'] == "{'model': 'maml', 'aug': 'comb'}"] = "{'comb': 'rand', 'space': 'prop', 'weight': 0.5, 'factor': 10}"
means['head_params'][means['head_params'] == "{'model': 'maml', 'aug': 'comb'}"] = "{'model': 'maml'}"

means = means.round(2)

mat = []
for i in range(12):
    out = str(i+1)
    row = []
    for head in head_params:
        out += ' & '
        tmp_df = means[means['data_idx'] == i]
        tmp_df = tmp_df[tmp_df['head_params'] == head]
        val = tmp_df[metric][tmp_df['aug_params'] == y_idx[0]].values[0]
        val_aug = tmp_df[metric][tmp_df['aug_params'] == y_idx[1]].values[0]
        row.append(val)
        row.append(val_aug)
        if val == max(val, val_aug):
            out += '\\textbf{' + "{:.2f}".format(val) + '}'
        else:
            out += "{:.2f}".format(val)
        out += ' & '
        if val_aug == max(val, val_aug):
            out += '\\textbf{' + "{:.2f}".format(val_aug) + '}'
        else:
            out += "{:.2f}".format(val_aug)
    mat.append(row)
    out += '\\\\'
    print(out)

out = '\\midrule'
print(out)

mat = np.array(mat)
out = 'Mean'
for j in range(mat.shape[1]):
    out += ' & '
    if j % 2 == 1:
        # Can override manually whenever we see a tie
        out += '\\textbf{'
    out += "{:.2f}".format(np.mean(mat[:, j]))
    if j % 2 == 1:
        out += '}'

out += '\\\\'
print(out)
