from torch.utils.data import Dataset
import torch.distributions as dist
import torch
import math


def sample_gaussian_mixture(means, stds, weights, num_samples):
    k = len(weights)
    dim = means.shape[1]

    categorical = dist.Categorical(weights)

    mixture = dist.MixtureSameFamily(
        categorical, dist.Independent(dist.Normal(means, stds.expand(dim, k).T), 1)
    )

    samples = mixture.sample((num_samples,))
    return samples


class MixturedGaussianDataset(Dataset):
    def __init__(self, num_samples, i: int):
        self.num_samples = num_samples
        means1 = torch.tensor([[0, -1], [0, -3]])
        means2 = torch.tensor([[1 / math.sqrt(3), 1], [math.sqrt(3), 3]])
        means3 = torch.tensor([[-1 / math.sqrt(3), 1], [-math.sqrt(3), 3]])
        self.means = torch.stack([means1, means2, means3], dim=0).float()
        self.means = self.means[None, i,:]
        self.weights = torch.tensor([1] * 2).float()
        self.stds = torch.tensor([0.01, 0.05]).float()
        self.xdata = torch.cat(
            [
                sample_gaussian_mixture(means, self.stds, self.weights, num_samples)
                for means in self.means
            ],
            dim=0,
        )
        self.cdata = torch.ones(num_samples) * i
        assert len(self.xdata) == len(self.cdata)
        self.x_dim = 2
        self.c_dim = 1

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        return self.xdata[idx], self.cdata[idx]


class MultiMarginalMMD(Dataset):
    def __init__(self, num_samples):
        dataset0 = MixturedGaussianDataset(num_samples, 0)
        dataset1 = MixturedGaussianDataset(num_samples, 1)
        dataset2 = MixturedGaussianDataset(num_samples, 2)
        self.datas = [dataset0, dataset1, dataset2]

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return self.datas[idx]
class RotatingPokeball(Dataset):
    def __init__(self,num_samples,theta_in:int,K:int):
        super().__init__()
        self.num_samples = num_samples
        theta = torch.rand(num_samples,1) * 2*torch.pi
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        )* torch.tensor([[1], [2]])[r]
        samples[samples[:,1]>0] = samples[samples[:,1]>0] + torch.tensor([[0,1]])
        samples[samples[:,1]<=0] = samples[samples[:,1]<=0] + torch.tensor([[0,-1]])
        theta =  torch.rand(num_samples,1) * torch.pi
        rotation_matrix = torch.stack([torch.cat([torch.cos(theta), -torch.sin(theta)], dim=1),torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)],dim=1)
        samples = torch.einsum("bij,bj->bi",rotation_matrix,samples)
        classes = torch.bucketize(theta,torch.linspace(0,torch.pi,K+1))-1
        self.xdata = samples[(classes==theta_in).squeeze()]
        self.cdata = classes[classes==theta_in]
        self.x_dim = 2
        self.c_dim = 1
    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        return self.xdata[idx], self.cdata[idx]
class MultiMarginalRotatingPokeball(Dataset):
    def __init__(self, num_samples,K):
        self.datas = [RotatingPokeball(num_samples,k,K) for k in range(K)]
        self.x_dim = 2
        self.c_dim = 1

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

class CircleMixturedGaussianDataset(Dataset):

    def __init__(self, num_samples,K,k, theta_max=math.pi * 2):
        super().__init__()
        self.num_samples = num_samples
        theta = torch.rand(num_samples, 1) * theta_max
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        ) * torch.tensor([[1], [2]])[r]
        samples = samples + torch.randn_like(samples) * 0.01
        classes = torch.clip(torch.bucketize(theta,torch.linspace(0,theta_max,K+1))-1,min=0)
        self.xdata = samples[(classes==k).squeeze()]
        self.cdata = classes[classes==k]
        self.x_dim = 2
        self.c_dim = 1
    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        return self.xdata[idx], self.cdata[idx]
class MultiMarginalCircleMMD(Dataset):
    def __init__(self, num_samples,K):
        self.datas = [CircleMixturedGaussianDataset(num_samples,K,k,theta_max=math.pi) for k in range(K)]
        self.x_dim = 2
        self.c_dim = 1

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

class CircleMixturedFilledGaussianDataset(Dataset):

    def __init__(self, num_samples,K,k, theta_max=math.pi /2):
        super().__init__()
        self.num_samples = num_samples
        theta = torch.rand(num_samples, 1) * theta_max
        r = torch.rand(num_samples,1)+1
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        ) * r
        classes = torch.clip(torch.bucketize(theta,torch.linspace(0,theta_max,K+1))-1,min=0)
        self.xdata = samples[(classes==k).squeeze()]
        self.cdata = classes[classes==k]
        self.x_dim = 2
        self.c_dim = 1
    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        return self.xdata[idx], self.cdata[idx]
class MultiMarginalFilledCircleMMD(Dataset):
    def __init__(self, num_samples,K):
        self.datas = [CircleMixturedFilledGaussianDataset(num_samples,K,k,theta_max=math.pi/2) for k in range(K)]
        self.x_dim = 2
        self.c_dim = 1

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]