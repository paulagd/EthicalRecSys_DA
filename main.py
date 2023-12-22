import torch
from IPython import embed
from utils import parse_args
from datasets import load_data
from models import FactorizationMachineModel, ItemPop, RandomModel
from train import train_and_test
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':

    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    train_dataset, rating_mat, dims, data_loader, val_loader, test_loader = load_data(args)

    #########################################################
    # SELECT MODEL AND PROCEED
    #########################################################
    if args.model == 'fm':
        model = FactorizationMachineModel(train_dataset.dims, args.k).to(device)
    elif args.model == 'itempop':
        args.inference = True
        model = ItemPop(train_dataset, dims, args.topk).to(device)
    elif args.model == 'random':
        args.inference = True
        model = RandomModel(dims, args.topk).to(device)
    else:
        raise NotImplementedError('Model not implemented yet.')

    train_and_test(args, dims, model, data_loader, device, val_loader, test_loader)
