import torch.utils.data as data
import glob

from IPython import embed
from random import sample
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from utils import *
from metrics import get_merit_dict
from split import get_numpy_data, split_train_test

                                     
def load_data(args):

    df = load_ml_1m_dataset(args.dataset)

    if type(df) == list:
        train_x, val_x, dims = df
        test_x = val_x
    else:
        data, dims = get_numpy_data(df, ['user', 'item', 'timestamp'])
        train_x, test_x, val_x = split_train_test(data, dims, args.dataset)
        dims = dims[:2]

    print(f'#users: {dims[0]}')
    print(f'#items: {dims[1] - dims[0]}')
    print(f'#interactions: {len(train_x)}')
    print(f'#VAL interactions: {len(val_x)}')
    print(f'#TEST interactions: {len(test_x)}')

    sequential = False
    flag = '_popularity_sampling_' if args.popsampling else ''
    tr_path = f'data/{args.dataset}/preprocessed_data/dataset{flag}_seq={int(sequential)}_data_dims={dims[-1]}_negs={args.num_ng}'
    train_dataset, rating_mat = get_train_dataset(train_x, dims, args, tr_path)
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.test:
        val_loader = None
        test_x = get_eval_set(f'data/{args.dataset}/preprocessed_data/testx_logged.pkl', test_x, train_x, dims, args)
        test_dataset = TestData(test_x, dims, train_dataset, tr_path=tr_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)#,
    else:
        test_loader = None
        val_x = get_eval_set(f'data/{args.dataset}/preprocessed_data/valx_logged.pkl', val_x, train_x, dims, args)
        val_dataset = TestData(val_x, dims, train_dataset, tr_path=tr_path)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size if args.cand > 0 else 1, shuffle=False,
                                num_workers=args.num_workers)

    return train_dataset, rating_mat, dims, data_loader, val_loader, test_loader


class PointData(data.Dataset):
    def __init__(self, train_x, dims, num_ng, pop_merits, path, train_flag=True):
        """
        Dataset formatter adapted point-wise algorithms
        Parameters
        """
        super(PointData, self).__init__()
        self.dims = dims
        if not train_flag:
            path = f'{path}_inference'

        self.path = path if not glob.glob(f'{path}.*') else glob.glob(f'{path}.*')[0]
        if os.path.exists(self.path):
            self.interactions = np.load(self.path)
        else:
            self.interactions = self.__get_sampled_data__(train_x, num_ng, pop_merits)

    def __get_sampled_data__(self, data, num_ng, pop_merits):

        interactions = []
        flag = 'popular' if pop_merits else 'uniform'
        for u in trange(self.dims[0], desc=f'Sampling from {flag} distribution data...'):
            user_data = data[data[:, 0] == u]
            negs = ng_sample(user_data[:, 1], self.dims, num_ng, pop_merits)
            interactions.append(np.column_stack((user_data, np.tile(1, len(user_data)))))
            interactions.append(np.column_stack((np.tile(u, len(negs)), negs, np.tile(0, len(negs)))))

        interactions = np.vstack(interactions)
        np.save(self.path, interactions, allow_pickle=True,fix_imports=True)
        return interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        """
        Return the pairs user-item and the target.
        """
        return self.interactions[index][:-1], self.interactions[index][-1]


def get_train_dataset(train_x, dims, args, path):
    rating_mat = build_adj_mx(dims[-1], train_x, args.dataset)
    pop_merits = None if not args.popsampling else list(get_merit_dict(list(train_x[:, 1]), dims).values())
    train_dataset = PointData(train_x, dims, args.num_ng, pop_merits, path)
    return train_dataset, rating_mat


class TestData(data.Dataset):
    def __init__(self, test_x,  dims, train_dataset, sequential=False, tr_path=None):
        """
        Dataset formatter adapted to test
        """
        super(TestData, self).__init__()

        self.test_x = test_x
        self.dims = dims
        self.sequential = sequential
        if self.sequential:
            self.user_seq = train_dataset.seq
        self.tr_path = tr_path if not glob.glob(f'{tr_path}.*') else glob.glob(f'{tr_path}.*')[0]
        if os.path.exists(self.tr_path):
            self.train_interactions = np.load(self.tr_path)
            pos_train = self.train_interactions[self.train_interactions[:, -1] == 1][:, :2]
            tr_items_per_user = np.split(pos_train[:,1], np.unique(pos_train[:, 0], return_index=True)[1][1:])
            pop = get_merit_dict(np.concatenate(tr_items_per_user), dims, get_abs_val=True)
            sorted_pop = sorted(pop.items(), key=lambda x:x[1], reverse=True)
            np.save(f"data/{tr_path.split('/')[1]}/preprocessed_data/sorted_pop.npy", sorted_pop)
            self.tr_pop_users, self.clusters = get_pop_users(tr_items_per_user,
                                                             np.load(f"data/{tr_path.split('/')[1]}/preprocessed_data/sorted_pop.npy"),
                                                             dims[0], tr_path.split('/')[1])
        else:
            raise NotImplementedError('Need to be saved the training data first')

    def __len__(self):
        return len(self.test_x)

    def __getitem__(self, index):
        """
        Return the pairs user-GTitem and candidates.
        """
        if self.sequential:
            return self.test_x[index][:, 1], self.user_seq[index]
        else:
            return self.test_x[index], []


def get_eval_set(testfile, test_x, train_x, dims, args):
    os.makedirs(os.path.dirname(testfile), exist_ok=True)

    if not os.path.exists(testfile):
        if not os.path.exists(os.path.join(os.path.dirname(testfile), 'GT_per_users.pkl')):
            gt_users = {}
            for u in trange(dims[0], desc = 'creating dict of GT users'):
                gt_users[u] = train_x[train_x[:, 0] == u][:, 1]

            with open(os.path.join(os.path.dirname(testfile), 'GT_per_users.pkl'), "wb") as fp:   #Pickling
                pickle.dump(gt_users, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(os.path.dirname(testfile), 'GT_per_users.pkl'), 'rb') as handle:
                gt_users = pickle.load(handle)
                # np.load(self.path)
        test_set = []
        all_set = set(range(dims[0], dims[1]))
        for u in tqdm(range(dims[0]), desc="BUILDING EVAL SET..."):
            gt_pair = test_x[test_x[:, 0] == u][0]
            negatives = all_set - set(gt_users[u]) - set([gt_pair[1].copy()])

            single_user_test_set = np.vstack([gt_pair, ] * (len(negatives)+1))
            single_user_test_set[:, 1][1:] = np.array(list(negatives))
            test_set.append(single_user_test_set.copy())
        
        with open(testfile, "wb") as fp:   #Pickling
            pickle.dump(test_set, fp)
    else:
        with open(testfile, "rb") as fp:   # Unpickling
            test_set = pickle.load(fp)

    if args.cand != 0:
        new_set = []
        flag = 'popularity' if args.test_popsampling else 'uniform'
        pop_merits = None if not args.test_popsampling else list(get_merit_dict(list(train_x[:, 1]), dims).values())

        for user_set in tqdm(test_set, desc=f'{flag} TEST sampling for inference ...'):
            if args.test_popsampling and pop_merits:
                cand = list(user_set[:, 1][1:]) - dims[0]
                assert len(pop_merits) == dims[1]-dims[0]
                pop_cand = np.take(pop_merits, cand)
                idx = np.random.choice(len(pop_cand), args.cand, p=list(pop_cand/sum(pop_cand)), replace=False)
                new_neg = list(np.take(cand, idx) + dims[0])
            else:
                new_neg = sample(list(user_set[:, 1][1:]), min(args.cand, len(list(user_set[:, 1][1:]))))

            new_set.append(np.column_stack((user_set[:, 0][:args.cand+1], [user_set[0][1]] + new_neg)))
        test_set = new_set

    return test_set