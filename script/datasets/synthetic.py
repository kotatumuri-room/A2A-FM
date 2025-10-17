from torch.utils.data import Dataset
from numpy import ndarray
from omegaconf import DictConfig
import numpy as np
import torch.distributions as dist
import torch
import math
from script.datasets.base_dataset import BaseDataset


def sample_gaussian_mixture(means, stds, weights, num_samples):
    k = len(weights)
    dim = means.shape[1]

    categorical = dist.Categorical(weights)

    mixture = dist.MixtureSameFamily(
        categorical, dist.Independent(dist.Normal(means, stds.expand(dim, k).T), 1)
    )

    samples = mixture.sample((num_samples,))
    return samples


class MixturedGaussianDataset(BaseDataset):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        means1 = torch.tensor([[0, -1], [0, -3]])
        means2 = torch.tensor([[ math.sqrt(3)/2, 1], [math.sqrt(3)/2*3, 3]])
        means3 = torch.tensor([[- math.sqrt(3)/2, 1], [-math.sqrt(3)/2*3, 3]])
        self.means = torch.stack([means1, means2, means3], dim=0).float()
        self.weights = torch.tensor([1] * 2).float()
        self.stds = torch.tensor([0.01, 0.05]).float()
        self.xdata = torch.cat(
            [
                sample_gaussian_mixture(means, self.stds, self.weights, num_samples)
                for means in self.means
            ],
            dim=0,
        )
        self.cdata = torch.cat(
            [torch.tensor([i for _ in range(num_samples)]) for i in (0, 1, 2)], dim=0
        )
        self.x_dim = 2
        self.c_dim = 1


class CircleMixturedGaussianDataset(BaseDataset):

    def __init__(self, num_samples, theta_max=math.pi * 2, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        theta = torch.rand(num_samples, 1) * theta_max
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        ) * torch.tensor([[1], [2]])[r]
        samples = samples + torch.randn_like(samples) * 0.01
        self.xdata = samples
        self.cdata = theta
        self.x_dim = 2
        self.c_dim = 1
    def sample(self,theta,num_samples):
        theta = torch.ones(num_samples,1)*theta
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        ) * torch.tensor([[1], [2]])[r]
        samples = samples + torch.randn_like(samples) * 0.01
        return samples
class CircleMixturedFilledGaussianDataset(BaseDataset):

    def __init__(self, num_samples, theta_max=math.pi/2, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        theta = torch.rand(num_samples, 1) * theta_max
        r = torch.rand(num_samples,1) + 1
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        ) * r
        self.xdata = samples
        self.cdata = theta
        self.x_dim = 2
        self.c_dim = 1
    def sample(self,theta,num_samples):
        theta = torch.ones(num_samples,1)*theta
        r = torch.rand(num_samples,1) + 1
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        ) * r
        return samples

class RotatingPokeball(BaseDataset):
    def __init__(self,num_samples,noise_intensity=0,**kwargs):
        super().__init__(**kwargs)
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
        samples = samples + torch.randn_like(samples) * noise_intensity
        self.noise_intensity = noise_intensity
        self.xdata = samples
        self.cdata = theta
        self.x_dim = 2
        self.c_dim = 1
    def sample(self,theta_in,num_samples):
        theta = torch.rand(num_samples,1) * 2*torch.pi
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        )* torch.tensor([[1], [2]])[r]
        samples[samples[:,1]>0] = samples[samples[:,1]>0] + torch.tensor([[0,1]])
        samples[samples[:,1]<=0] = samples[samples[:,1]<=0] + torch.tensor([[0,-1]])
        theta =  torch.ones(num_samples,1) * theta_in
        rotation_matrix = torch.stack([torch.cat([torch.cos(theta), -torch.sin(theta)], dim=1),torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)],dim=1)
        samples = torch.einsum("bij,bj->bi",rotation_matrix,samples)
        samples = samples + torch.randn_like(samples) * self.noise_intensity
        return samples
class RotatingPokeballThree(BaseDataset):
    def __init__(self,num_samples,**kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        theta = torch.rand(num_samples,1) * 2*torch.pi
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        )* torch.tensor([[1], [2]])[r]
        samples[samples[:,1]>0] = samples[samples[:,1]>0] + torch.tensor([[0,1]])
        samples[samples[:,1]<=0] = samples[samples[:,1]<=0] + torch.tensor([[0,-1]])
        theta =  torch.multinomial(torch.tensor([0,torch.pi/4,torch.pi/2]),num_samples,replacement=True).reshape(-1,1)
        rotation_matrix = torch.stack([torch.cat([torch.cos(theta), -torch.sin(theta)], dim=1),torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)],dim=1)
        samples = torch.einsum("bij,bj->bi",rotation_matrix,samples)
        # samples = samples + torch.randn_like(samples) * 0.01
        self.xdata = samples
        self.cdata = theta
        self.x_dim = 2
        self.c_dim = 1
    def sample(self,theta_in,num_samples):
        theta = torch.rand(num_samples,1) * 2*torch.pi
        r = torch.multinomial(torch.tensor([1.0, 1.0]), num_samples, replacement=True)
        samples = (
            torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        )* torch.tensor([[1], [2]])[r]
        samples[samples[:,1]>0] = samples[samples[:,1]>0] + torch.tensor([[0,1]])
        samples[samples[:,1]<=0] = samples[samples[:,1]<=0] + torch.tensor([[0,-1]])
        theta =  torch.ones(num_samples,1) * theta_in
        rotation_matrix = torch.stack([torch.cat([torch.cos(theta), -torch.sin(theta)], dim=1),torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)],dim=1)
        samples = torch.einsum("bij,bj->bi",rotation_matrix,samples)
        return samples