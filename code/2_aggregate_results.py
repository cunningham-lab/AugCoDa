
import os
import pandas as pd

# Read in results and aggregate into summaries
res_dir = './out/mlrepo12/'
res = []
tables = os.listdir(res_dir)
for table in tables:
    res.append(pd.read_csv(res_dir + table, index_col=0))

res_dir = './out/mlrepo12maml/'
tables = os.listdir(res_dir)
for table in tables:
    res.append(pd.read_csv(res_dir + table, index_col=0))

res_dir = './out/mlrepo12deepmicro/'
tables = os.listdir(res_dir)
for table in tables:
    res.append(pd.read_csv(res_dir + table, index_col=0))

res_dir = './out/mlrepo12contrastive/'
tables = os.listdir(res_dir)
for table in tables:
    res.append(pd.read_csv(res_dir + table, index_col=0))

res = pd.concat(res)

res.to_csv('./out/mlrepo12.csv')