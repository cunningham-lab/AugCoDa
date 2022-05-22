
# %%
import numpy as np
import pandas as pd

# %%
df1 = pd.read_csv('./out/mlrepo12.csv')

# %%
metric = 'auc'
tran_param = "{'space': 'prop'}"


head_params = [
    "{'model': 'contrastive', 'head': 'lp', 'init': 'rand'}",
    "{'model': 'deepmicro', 'head': 'lp'}",
    "{'model': 'contrastive', 'head': 'lp'}",
    "{'model': 'contrastive', 'head': 'ft', 'init': 'rand'}",
    "{'model': 'deepmicro', 'head': 'ft'}",
    "{'model': 'contrastive', 'head': 'ft'}",
]

head_params = [
    "{'model': 'contrastive', 'head': 'lp', 'init': 'rand'}",
    "{'model': 'deepmicro', 'head': 'lp'}",
    "{'model': 'contrastive', 'head': 'lp', 'aug': 'am'}",
    "{'model': 'contrastive', 'head': 'ft', 'init': 'rand'}",
    "{'model': 'deepmicro', 'head': 'ft'}",
    "{'model': 'contrastive', 'head': 'ft', 'aug': 'am'}",
]

head_params = [
    "{'model': 'contrastive', 'head': 'lp', 'init': 'rand'}",
    "{'model': 'deepmicro', 'head': 'lp'}",
    "{'model': 'contrastive', 'head': 'lp', 'aug': 'cc'}",
    "{'model': 'contrastive', 'head': 'ft', 'init': 'rand'}",
    "{'model': 'deepmicro', 'head': 'ft'}",
    "{'model': 'contrastive', 'head': 'ft', 'aug': 'cc'}",
]

# average over seeds
means = df1.pivot_table(
    values=[metric], 
    index=['data_idx', 'aug_params', 'tran_params', 'dr_params', 'head_params'], 
    aggfunc='std'
) / np.sqrt(20)
means = means.reset_index()
means = means[means['tran_params'] == tran_param]

means = means[means['head_params'].isin(head_params)]

out = means.pivot_table(values=[metric], index=['data_idx'], columns=['head_params'], aggfunc='mean')

print(out)

print(out.mean())

means = means.round(2)

mat = []
for i in range(12):
    out = str(i+1)
    row = []
    for j in [0, 3]:
        out += ' & '
        tmp_df = means[means['data_idx'] == i]
        val1 = tmp_df[metric][tmp_df['head_params'] == head_params[j]].values[0]
        val2 = tmp_df[metric][tmp_df['head_params'] == head_params[j + 1]].values[0]
        val3 = tmp_df[metric][tmp_df['head_params'] == head_params[j + 2]].values[0]

        row.append(val1)
        row.append(val2)
        row.append(val3)
        if val1 == max(val1, val2, val3):
            out += '\\textbf{' + "{:.2f}".format(val1) + '}'
        else:
            out += "{:.2f}".format(val1)
        out += ' & '
        if val2 == max(val1, val2, val3):
            out += '\\textbf{' + "{:.2f}".format(val2) + '}'
        else:
            out += "{:.2f}".format(val2)
        out += ' & '
        if val3 == max(val1, val2, val3):
            out += '\\textbf{' + "{:.2f}".format(val3) + '}'
        else:
            out += "{:.2f}".format(val3)
    mat.append(row)
    out += '\\\\'
    print(out)

out = '\\midrule'
print(out)

mat = np.array(mat)
out = 'Mean'
for j in range(mat.shape[1]):
    out += ' & '
    if j % 3 == 2:
        # can override manually when we have a tie
        out += '\\textbf{'
    out += "{:.2f}".format(np.mean(mat[:, j]))
    if j % 3 == 2:
        out += '}'

out += '\\\\'
print(out)