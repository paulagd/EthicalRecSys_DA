import natsort
import torch

from tqdm import trange, tqdm
from IPython import embed
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import Adam

from metrics import getHitRatio, getNDCG, get_coverage, get_fairness_metric, calc_diversity
from utils import *
from models import *


"""#### **Train**"""
def train_one_epoch(dims, model, data_loader, criterion, device, optimizer):
    model.train()
    flag = 'training'
    total_loss = []
    for i, loader in enumerate(tqdm(data_loader, desc=f"{flag}...")):
        interactions, targets = loader[0].to(device), loader[1].to(device)
        logits = model(interactions)
        loss = criterion(logits, targets.float())

        optimizer.zero_grad()
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()

    return np.mean(total_loss)


"""#### **Inference**"""
def test(model, test_loader, device, dims, args, training=False):
    # Test the HR and NDCG for the model @topK
    flag = '_popsampling' if args.popsampling else ''
    sampler = '_RULETA' if args.sampler else ''
    print(f'flag: data/{args.dataset}_full_rank_items_{args.model}{flag}{sampler}.npy')
    model.eval()
    HR, NDCG, rec_items, full_rank_items, gt_items_all, HR_decomposed, all_logits = [], [], [], [], [], [], []
    user_diversity = []
    for user_test, user_seq in tqdm(test_loader, desc="inference..."):
        if args.model == 'random' or args.model == 'itempop':
            gt_items = user_test[:, :, 1][:, 0]
            predictions = model.predict(user_test, device=device)
            recommend_list = predictions.int()
            
        else:
            gt_items = user_test[:, :, 1][:, 0]
            gt_items_all.append(gt_items)
            logits = model.predict(user_test.reshape(-1, len(dims)), device)
            logits = logits.reshape(user_test.shape[:2])
            if args.sampler:
                if args.cand == 0:
                    shorten = 5000
                    # print(f'Shorten: {shorten}')
                    _, indices = torch.topk(logits, shorten)
                    logits = torch.gather(logits, 1, indices)

                preds = np.stack([gumbel_sampling(l.unsqueeze(0).cpu().detach(), 1, args.topk)[0].numpy() for l in logits])
                if args.cand == 0:
                    # IDEA: sort the shorten predictions
                    preds = np.take_along_axis(indices.cpu().detach(), preds, axis=1).numpy()
                recommend_list = np.take_along_axis(user_test[:, :, 1], preds, axis=1)
            else:
                _, indices = torch.topk(logits, args.topk)
                recommend_list = torch.gather(user_test[:, :, 1], 1, indices.cpu().detach())

            if not training:
                values, indices = torch.topk(logits, args.topk)
                if args.model == 'neufm':
                    user_diversity.append(calc_diversity(model.embedding_MLP.weight[indices[0]]))
                else:
                    user_diversity.append(calc_diversity(model.embedding.weight[indices[0]]))

        rec_items.append(np.stack(recommend_list))
        HR_decomposed.append([gt_items[0] if gt_items[0] in recommend_list[0] else -1])
        HR.append([getHitRatio(r, gt) for r, gt in zip(recommend_list, gt_items)])
        NDCG.append([getNDCG(r, gt) for r, gt in zip(recommend_list, gt_items)])

    assert args.topk == len(recommend_list[0])
    if not training:
        cov, top_items = get_coverage(np.concatenate(rec_items), dims[1]-dims[0], args.topk)
        return np.mean(np.concatenate(HR)), np.mean(np.concatenate(NDCG)), cov, rec_items, np.mean(user_diversity)
    else:
        return np.mean(np.concatenate(HR)), np.mean(np.concatenate(NDCG)), 0, rec_items, 0


"""#### **Train and Test**"""
def train_and_test(args, dims, model, data_loader, device, val_loader, test_loader):
    popsampling = '_popsampling' if args.popsampling else ''
    sampler = '_sampler_' if args.sampler else ''
    weights_folder = f'weights_logged/{args.dataset}/{popsampling}'

    if args.inference or args.test:
        if os.path.exists(weights_folder) and not (args.model in ['random', 'itempop']):
            output_names = [name for name in os.listdir(weights_folder) if name.startswith(f'{args.model}_')]
            last_weight_saved = natsort.natsorted(output_names,reverse=True)[0]
            checkpoint = torch.load(os.path.join(weights_folder, last_weight_saved))
            model.load_state_dict(checkpoint['state_dict'])
            print(f'Loaded weights from {os.path.join(weights_folder, last_weight_saved)} !')
        else:
            print('NO weights loaded!')

        if args.test:
            flag = 'TEST'
            hr, ndcg, cov, rec_topk_list, diversity = test(model, test_loader, device, dims, args)
            loader = test_loader
        else:
            flag = 'VAL'
            hr, ndcg, cov, rec_topk_list, diversity = test(model, val_loader, device, dims, args)
            loader = val_loader

        np.save(f'data/{args.dataset}/preprocessed_data/top_rec_list_{args.model}{popsampling}{sampler}.npy',
                np.concatenate(rec_topk_list), allow_pickle=True)
        eoe, novelty, arp = get_fairness_metric(data_loader.dataset.interactions, dims, rec_topk_list, loader, args=args)
        print(f' {flag}: HR@{args.topk} = {hr:.4f}, NDCG@{args.topk} = {ndcg:.4f}, cov@{args.topk} = {cov:.4f},'
              f' EoE@{args.topk} = {eoe:.4f}, div@{args.topk} = {diversity:.4f} , nov@{args.topk} = {novelty:.4f},'
              f'arp@{args.topk} = {arp:.4f}')
    else:
        criterion = BCEWithLogitsLoss(reduction='mean')
        optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)

        best_HR, best_ndcg, best_epoch = 0, 0, 0
        stop = 0
        for epoch_i in trange(args.epochs):
            train_loss = train_one_epoch(dims, model, data_loader, criterion, device, optimizer)
            hr, ndcg, cov, rec_topk_list, diversity = test(model, val_loader, device, dims, args, training=True)

            print(f'epoch {epoch_i}:')
            print(f'training loss = {train_loss:.4f} ')
            print(f'Eval: HR@{args.topk} = {hr:.4f}, NDCG@{args.topk} = {ndcg:.4f}')

            os.makedirs(weights_folder, exist_ok=True)
            if best_HR > hr:
                if stop == 10:
                    print(f'best epoch = {best_epoch} | BEST HR@{args.topk} = {best_HR:.4f}, BEST NDCG@{args.topk} = {best_ndcg:.4f}')
                    exit()
                stop +=1
            else:
                stop = 0
                best_HR = hr
                best_ndcg = ndcg
                best_epoch = epoch_i

            if stop == 0:  # save just when its better
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(weights_folder, f'{args.model}_weights_epoch={epoch_i}.pkl'))

        print(f'best epoch = {best_epoch} | BEST HR@{args.topk} = {best_HR:.4f}, BEST NDCG@{args.topk} = {best_ndcg:.4f}')

