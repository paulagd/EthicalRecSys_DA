import math
import numpy as np
import pandas as pd
import pickle
import torch
from collections import Counter
from tqdm import tqdm

from IPython import embed


def getHitRatio(recommend_list, gt_item):
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    if gt_item in recommend_list:
        index = np.where(gt_item == recommend_list)[0][0]
        return np.reciprocal(np.log2(index+2))
    return 0


def get_coverage(pred_items, n_items, k=None):
    """
    Coverage --> analyses the diversity among recommended items
    """
    if k:
        top_items = np.concatenate([items[:k] for items in pred_items])
    else:
        top_items = np.concatenate(pred_items)
    return len(np.unique(top_items)) / n_items, top_items


def get_merit_dict(itemslist, dims, get_abs_val=False, sum_merits=False):
    items, freq = np.unique(itemslist, return_index=False, return_counts=True)
    item_merits = {k: 0 for k in list(np.arange(dims[0], dims[1]))}   # necessary for filling with 0 the elements with no merits
    for item, pop in zip(items, freq):
        item_merits[item] = pop
    if sum_merits:
        return sum([v/sum(item_merits.values()) for v in item_merits.values()])
    else:
        return item_merits if get_abs_val else {k:v/sum(item_merits.values()) for k,v in item_merits.items()}


def get_fairness_metric(train_interactions, dims, recommended_items, val_set, args=None):
    positive_interactions = train_interactions[train_interactions[:, -1] == 1]
    assert positive_interactions[:, -1].all() == 1, "All labels must be positive to compute EoE"
    tr_items =  positive_interactions[:, 1]
    ######################################
    # # IDEA: ARP
    pop_items = np.load(f"data/{args.dataset}/preprocessed_data/sorted_pop.npy", allow_pickle=True)
    keys = pop_items[:, 0]
    values = pop_items[:, 1]/sum(pop_items[:, 1])
    recommended_items = np.vstack(recommended_items)
    merit_dict = {keys[i]: values[i] for i in range(len(keys))}
    arp = []
    for row in tqdm(recommended_items, desc='arp ...'):
        arp.append(np.sum([merit_dict[i] for i in row])/len(row))

    # IDEA: NOVELTY
    candidates = [a[1:][:, 1] for a in val_set.dataset.test_x]
    cand = Counter(np.concatenate(candidates))
    rec = Counter(np.concatenate(recommended_items))
    novelty_item = []
    for i in list(rec.keys()):
        novelty_item.append(1 - (rec[i]/cand[i]))
    novelty = np.mean(novelty_item)

    #######################################

    M = get_merit_dict(tr_items, dims)
    P = get_merit_dict(np.concatenate(recommended_items), dims)

    # IDEA: save those dicts to plot merits
    if args.savemerits:
        ru = '_sampler' if args.sampler else ''
        popsampling = '_popsampling' if args.popsampling else ''
        with open(f'M_{args.model}{ru}{popsampling}.pkl', 'wb') as f:
            pickle.dump(M, f)
        with open(f'P_{args.model}{ru}{popsampling}.pkl', 'wb') as f:
            pickle.dump(P, f)
        exit()

    a = pd.DataFrame([M,P]).T.reset_index()
    a.rename(columns={"index": "items", 0: "M", 1:"P"}, inplace=True)
    pond_foe = np.abs(a.P.fillna(0) - a.M.fillna(0))
    return np.sum(pond_foe), novelty, np.mean(arp)


def calc_diversity(embeddings):
    final = []
    sym_count = 0
    for i in range(len(embeddings)):
        ans = []
        for j in range(len(embeddings)):
            if j >= sym_count:
                similarity = torch.cosine_similarity(embeddings[i].view(1,-1),
                                                     embeddings[j].view(1,-1)).item()
                ans.append(similarity)
        sym_count += 1
        final.append(ans)
    return np.sum(np.concatenate(final)) / len(np.concatenate(final))
