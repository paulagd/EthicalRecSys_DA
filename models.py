import torch, random
import numpy as np

from IPython import embed
from metrics import get_merit_dict


class ItemPop(torch.nn.Module):
    def __init__(self, train_dataset, dims, topk):
        super(ItemPop, self).__init__()

        """
        Simple popularity based recommender system
        """
        positive_interactions = train_dataset.interactions[train_dataset.interactions[:, -1] == 1]
        self.merit_dict = get_merit_dict(positive_interactions[:, 1], dims, get_abs_val=True)
        self.topk = topk

    def forward(self):
        pass

    def predict(self, interactions, device=None):
        assert interactions.shape[0] == 1, "need to be in test mode or batch_size=1"
        merits = {i.item(): self.merit_dict[i.item()] for i in interactions[0][:, 1]}
        idx_sorted = np.asarray(sorted(merits.items(), key=lambda x: x[1], reverse=True)[:self.topk])[:, 0]
        return torch.Tensor(idx_sorted).unsqueeze(0)

class RandomModel(torch.nn.Module):
    def __init__(self, dims, topk):
        super(RandomModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.topk = topk
        self.dims = dims

    def forward(self):
        pass

    def predict(self, interactions, device=None):
        assert interactions.shape[0] == 1, "need to be in test mode or batch_size=1"
        rand_list = random.sample(list(interactions[0][:,1].cpu().detach().numpy()), self.topk)
        return torch.Tensor(rand_list).unsqueeze(0)


class FM_operation(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = torch.nn.Linear(len(field_dims), 1)
        self.embedding = torch.nn.Embedding(field_dims[-1], embed_dim)
        torch.nn.init.xavier_normal_(self.embedding.weight)

        self.dims = field_dims
        self.fm = FM_operation(reduce_sum=True)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        embeddings = self.embedding(interaction_pairs)
        out = self.linear(interaction_pairs.float()) + self.fm(embeddings)
        y_pred = out.squeeze(1)
        return y_pred

    def predict(self, interactions, device):
        return self.forward(interactions.long().to(device))
