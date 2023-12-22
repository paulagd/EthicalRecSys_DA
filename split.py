import os.path
import numpy as np

from tqdm import trange
from IPython import embed


def get_numpy_data(df, cols=None):
    assert cols[-1] == 'timestamp'
    data = df[cols].astype('int32').to_numpy()
    add_dims=0
    for i in range(data.shape[1] - 1):  # do not affect to timestamp
        # MAKE IT START BY 0
        data[:, i] -= np.min(data[:, i])
        # RE-INDEX
        data[:, i] += add_dims
        add_dims = np.max(data[:, i]) + 1
    dims = np.max(data, axis=0) + 1

    return data, dims


def split_train_test(data, dims, dataset='ml-1m'):
    path = f'data/{dataset}/preprocessed_data/train_test_split_{dims[1]}.npz'
    os.makedirs(f'data/{dataset}/preprocessed_data/', exist_ok=True)
    if os.path.exists(path):
        data = np.load(path)
        return data['train_x'], data['test_x'], data['val_x']

    # Split and remove timestamp
    train_x, test_x, val_x = [], [], []
    for u in trange(dims[0], desc='spliting train/val/test and removing timestamp...'):
        user_data = data[data[:, 0] == u]
        sorted_data = user_data[user_data[:, -1].argsort()]
        if len(sorted_data) == 1:
            train_x.append(sorted_data[0][:-1])
        else:
            train_x.append(sorted_data[:-2][:, :-1])
            test_x.append(sorted_data[-1][:-1])
            val_x.append(sorted_data[-2][:-1])
    data = np.vstack(train_x), np.stack(test_x), np.stack(val_x)
    np.savez(path, train_x=data[0], test_x=data[1], val_x=data[2])
    return data
