import numpy as np
import pandas as pd

traindata_names = ["single_all_sparse510.npy", "single_all_sparse1020.npy", "single_all_sparse2050.npy",
                       "single_all_sparse50100.npy", "single_all_sparse100150.npy", "single_all_sparse278.npy"]
for name in traindata_names:
    s_data = np.load(name)
    sparse_samples = s_data[:, 0]
    df_sparse = pd.DataFrame(sparse_samples)
    eff_num2 = 288 - df_sparse.isna().sum(axis=1).values
    print(eff_num2)
    print(np.mean(eff_num2))
    print(np.min(eff_num2), np.max(eff_num2))