import pandas as pd
import os #, pickle
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import random
import pickle

from tqdm import tqdm, trange
from sklearn import preprocessing
from pandas.api.types import CategoricalDtype
from IPython import embed
from collections import Counter


def parse_args():

    parser = argparse.ArgumentParser(description='RECOMMENDER PARAMS')
    # common settings
    parser.add_argument('--seed', type=int, default=1234, help='pre-fixed seed for experiments')
    parser.add_argument('--model', type=str, default='fm', help='model to select: [fm, random, itempop]')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset to select: [ml-1m]')
    parser.add_argument('--loss', type=str, default='bcelogits', help='model to select: [bce, bcelogits]')

    parser.add_argument('--popsampling', action="store_true", default=False, help='whether apply a categorical sampler')
    parser.add_argument('--test_popsampling', action="store_true", default=False, help='whether apply a categorical sampler at test time')
    parser.add_argument('--sampler', action="store_true", default=False, help='whether apply a categorical sampler')

    parser.add_argument('--inference', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--savemerits', action="store_true", default=False)
    parser.add_argument('--topk', type=int, default=10, help='top number of recommend list')
    parser.add_argument('--num_ng', type=int, default=4, help='negative sampling number')
    parser.add_argument('--cand', type=int, default=0, help='candidates for test: if 0, do full-rank')
    parser.add_argument('--max_len', type=int, default=150, help='sequence of the user (context)')
    parser.add_argument('--save', action="store_true", default=False, help='activate to save weights')
    parser.add_argument('--statistics', action="store_true", default=False, help='activate to save statistics')

    # algorithm settings
    parser.add_argument('--k', type=int, default=64, help='latent factors numbers in the model')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    return args


def load_ml_1m_dataset(dataset):
    '''
    The dataset can be downloaded by doing
        - wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
        - unzip ml-1m.zip
    '''

    user, item = 'user', 'item'
    data_path = os.path.join('data', dataset)
    if not os.path.exists(os.path.join(data_path, "logged_interactions.pkl")):
        # users = pd.read_csv(os.path.join(data_path, "users.dat"), delimiter='::', engine='python',
        #                     names=['user', 'gender', 'age', 'occupation', 'zipcode'])

        df = pd.read_csv(os.path.join(data_path, "ratings.dat"), delimiter='::', engine='python',
                         names=[user, item, 'rating', 'timestamp'])

        # # Merge ratings & user features into one Data Frame via the 'user' column
        # df = rat.merge(users, on='user')
        df = df[df['rating']>=4]

        df = df[df[item].map(df[item].value_counts()) > 5]
        assert df[item].value_counts().values[-1] > 5

        df = df[df[user].map(df[user].value_counts()) > 5]
        assert df[user].value_counts().values[-1] > 5

        df[user] = pd.Categorical(df[user]).codes
        df[item] = pd.Categorical(df[item]).codes

        df.to_pickle(os.path.join(data_path, 'logged_interactions.pkl'))
    else:
        df = pd.read_pickle(os.path.join(data_path, 'logged_interactions.pkl'))

    return df[['user', 'item', 'timestamp']]


def build_adj_mx(n_feat, data, dataset='ml-1m'):
    path = f'data/{dataset}/preprocessed_data/adj_matrix_{n_feat}.npz'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        train_mat = sp.load_npz(path).todok()
    else:
        train_mat = sp.dok_matrix((n_feat, n_feat), dtype=np.float32)
        for x in tqdm(data, desc=f"BUILDING ADJACENCY MATRIX..."):
            train_mat[x[0], x[1]] = 1.0
            train_mat[x[1], x[0]] = 1.0

        sp.save_npz(path, train_mat.tocoo())
    return train_mat


def ng_sample(pos_items, dims, num_ng, pop_merits=None):
    min_item, max_item = dims[0], dims[1]
    # if num_ng is less than 20, it means we want to make "x" neg samples for each training sample
    num_ng = num_ng if num_ng > 20 else num_ng*len(pos_items)

    if pop_merits:
        # IDEA: Put items to avoid as 0 prob
        pos_items = pos_items - dims[0]
        pop_merits = [0 if i in pos_items else pop for i, pop in enumerate(pop_merits)]
        negs = min_item + np.random.choice(len(pop_merits), num_ng, p=list(pop_merits/sum(pop_merits)), replace=True)  # IDEA: Replace true?
    else:
        to_avoid = set(pos_items)
        total_items = set(range(min_item, max_item))
        to_sample_from = total_items.difference(to_avoid)
        # IDEA: sampling with replacement
        negs = np.random.choice(list(to_sample_from), size=num_ng, replace=True)
    return negs


def build_test_set(itemsnoninteracted, gt_test_interactions):
    # max_users, max_items = dims # number users (943), number items (2625)
    test_set = []
    for pair, negatives in tqdm(zip(gt_test_interactions, itemsnoninteracted), desc="BUILDING EVAL SET..."):
        # APPEND TEST SETS FOR SINGLE USER
        negatives = np.delete(negatives, np.where(negatives == pair[1]))
        single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
        single_user_test_set[:, 1][1:] = negatives
        test_set.append(single_user_test_set.copy())
    return test_set


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def gumbel_sampling(w, R, T):
    n = w.shape[1] # len(w)
    U = np.random.uniform(0,1,size=(R,n))
    G = w - np.log(- np.log(U))
    res = np.argsort(-G, axis=1)
    return res[:,:T]


def get_cluster(color):
    if color == 'teal':
        return 0
    elif color == 'mediumturquoise':
        return 1
    else:  # 'paleturquoise'
        return 2


def get_pop_vector_user(items, bins_dict, total_dict, bins=['teal', 'mediumturquoise', 'paleturquoise']):
    items_per_user = [bins_dict[i] for i in items]
    pop_users = Counter(items_per_user)
    a = {c:val/len(items) for c, val in pop_users.items()}
    for key in bins:
        if key not in a.keys():
            a[key] = 0

    return [a[bins[0]], a[bins[1]], a[bins[2]]]


def get_pop_users(consumed_items_per_usr, dict_item_color, n_users, dataset, train=True):
    if os.path.exists(f'data/{dataset}/preprocessed_data/pop_users.npy') and train:
        pop_users = np.load(f'data/{dataset}/preprocessed_data/pop_users.npy', allow_pickle=True)
    else:
        color_item = plot_barperpop(np.asarray(dict_item_color))
        reversed_dict = {}
        for item, c in color_item.items():
            if c not in reversed_dict:
                reversed_dict[c] = [item]
            else:
                reversed_dict[c].append(item)
        aux = {c:len(np.unique(l)) for c,l in reversed_dict.items()}

        pop_users = []
        for u in range(n_users):
            pop_vector = get_pop_vector_user(consumed_items_per_usr[u], color_item, aux)
            pop_users.append(pop_vector)

        pop_users = np.vstack(pop_users)
        if train:
            np.save(f'data/{dataset}/preprocessed_data/pop_users.npy', pop_users, allow_pickle=True)

    idx_sorted_users = np.lexsort((-pop_users[:, 1], -pop_users[:, 0]))
    sort_by_pop = pop_users[idx_sorted_users]

    clusters = plot_barperpop(np.vstack((idx_sorted_users, sort_by_pop[:, 0])).T, t1=0.6, t2=0.2,
                              normalize=False, title='sorted_pop_users_bins')

    return pop_users, clusters


def calc_mean_position_rank(full_rank_items):
    lst = list(full_rank_items)
    df = pd.DataFrame(lst).apply(pd.value_counts, axis=0)
    df2 = df.multiply(df.columns, axis=1)
    df3 = df2.sum(1)/df.sum(1)
    df3.index = df3.index.astype(int)
    d4 = df3.reset_index().rename(columns={"index": "items", 0: "order"})
    rank = d4.sort_values(by=['order'], ascending=True)['items'].values
    rank_value = d4.sort_values(by=['order'], ascending=True)['order'].values
    return rank, rank_value


def get_norm_hits_dict(hits, color_item, gt_counter=None):

    hr_dec = [color_item[k] for k in hits]
    clustered_real_hits = Counter(hr_dec)
    if gt_counter is None:
        all = sum(clustered_real_hits.values())
        final_perc_dict = []
        for key in ['teal', 'mediumturquoise', 'paleturquoise']:
            final_perc_dict.append(round(clustered_real_hits[key]/all * 100, 2))
    else:
        final_perc_dict = {}
        for key in ['teal', 'mediumturquoise', 'paleturquoise']:
            final_perc_dict[key] = round(clustered_real_hits[key]/gt_counter[key] * 100, 2)
    return final_perc_dict


def plot_barperpop(pop_per_items, t1=0.4, t2=0.2, title='pop_per_bin', normalize=True):

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    y = pop_per_items[:, 1] if not normalize else NormalizeData(pop_per_items[:,1])
    a = np.where(y >= t1, 'teal', np.where(t2 <= y, 'mediumturquoise', 'paleturquoise'))
    return dict(zip(pop_per_items[:, 0].astype(int), a))