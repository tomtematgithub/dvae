import numpy as np
import torch as t
from torch.nn import MSELoss
import torch.distributions as td
from torch.utils.data.sampler import SubsetRandomSampler
cuda = t.cuda.is_available()


eps = 1e-8


class Dataset(t.utils.data.Dataset):
    """
    This class creates the dataset for the convolutional model (3D data)
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X = self.data[index, :, :]

        return X


def get_dataset(training, validation, batch_size=128):
    """This method creates PyTorch dataloaders for training and
       vaidation sets to be used during training.

    Args:
        training (Torch dataset): training dataset
        validation (Torch dataset): validation dataset
        batch_size (int, optional): batch size. Defaults to 128.

    Returns:
        dataloaders: training and validation dataloaders
    """

    train = t.utils.data.DataLoader(training, batch_size=batch_size, pin_memory=cuda,
                                        sampler=SubsetRandomSampler(t.from_numpy(np.arange(len(training.data)))))
    valid = t.utils.data.DataLoader(validation, batch_size=batch_size, pin_memory=cuda,
                                        sampler=SubsetRandomSampler(t.from_numpy(np.arange(len(validation.data)))))

    return train, valid


def likelihood_loss(x, r, metric='BCE', loss='discrete'):
    """calculates likelihood loss between reconstructed and original data
    Args:
        x (tensor): original data
        r (tensor): reconstructed data

    Returns:
        tensor: likelihood loss between reconstructed and original data
    """
    # Combine axes 1 (input features) and 2 (time points)
    x = x.reshape(x.size()[0], -1)
    r = r.reshape(r.size()[0], -1)

    if metric == 'BCE':
        if loss == 'discrete':
            lh_loss = -t.sum(t.clamp(x, min=0, max=1) * t.log(t.clamp(r, min=eps, max=1)) +
                             t.clamp(1 - x, min=0, max=1) * t.log(t.clamp(1 - r, min=eps, max=1)), dim=1)
        elif loss == 'continuous':
            lh_loss = -t.sum(td.ContinuousBernoulli(probs=r).log_prob(x), dim=1)

    if metric == 'MSE':
        mse_loss = MSELoss(reduction='none')
        lh_loss = t.sum(mse_loss(x.float(), r), dim=-1)

    return lh_loss
