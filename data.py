import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List
from itertools import combinations


def wif(id):
    """
    Used to fix randomization bug for pytorch dataloader + numpy
    Code from https://github.com/pytorch/pytorch/issues/5059
    """
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


class MultitaskDataset():
    """
    Template class for a Multi-task data handler
    """
    def __init__(self) -> None:
        self.trainset: Dataset
        self.testset: Dataset

    def fetch_data_loaders(self, bs, workers=4, shuf=True) -> DataLoader:
        """
        Get the Dataloader for the entire dataset
        Args:
            - shuf       : Shuffle
            - wtd_loss   : Dataloader also has wts along with targets
            - wtd_sampler: Sample data from dataloader with weights
                         according to self.tr_wts
        """
        loaders = []
        for idx, data in enumerate([self.trainset, self.testset]):
            loaders.append(
                DataLoader(
                    data, batch_size=bs, shuffle=(idx==0) and shuf,
                    num_workers=workers, pin_memory=True,
                    worker_init_fn=wif))

        return loaders


class CifarDataset(MultitaskDataset):
    def split_dataset(self, task: List[int], permute: bool):
        """
        Use the "tasks" vector to split dataset
        """
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        for lab_id, lab in enumerate(task):
            task_tr_ind = np.where(np.isin(self.trainset.targets,
                                           [lab]))[0]
            task_te_ind = np.where(np.isin(self.testset.targets,
                                           [lab]))[0]

            # Get indices and store labels
            tr_ind.append(task_tr_ind)
            te_ind.append(task_te_ind)

            tr_lab.append([lab_id for _ in range(len(task_tr_ind))])
            te_lab.append([lab_id for _ in range(len(task_te_ind))])

        tr_ind, te_ind = np.concatenate(tr_ind), np.concatenate(te_ind)
        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        self.trainset.targets = list(tr_lab)
        self.testset.targets = list(te_lab)

        if permute:
            new_idx = np.random.permutation(len(self.trainset.targets))
            self.trainset.data = self.trainset.data[new_idx]
            self.trainset.targets = [self.trainset.targets[idx] for idx in new_idx]


class Cifar10Dataset(CifarDataset):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                 tasks: List[int],
                 download: bool = True,
                 permute: bool = True) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
        """
        train_transform, test_transform = self.get_transforms()

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=download,
            transform=test_transform)
        self.split_dataset(tasks, permute)

    def get_transforms(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        return train_transform, test_transform


class Cifar100Dataset(CifarDataset):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                 tasks: List[int],
                 download: bool = True,
                 permute: bool = True) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
        """
        train_transform, test_transform = self.get_transforms()

        self.trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=download,
            transform=test_transform)
        self.split_dataset(tasks, permute)

    def get_transforms(self):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        return train_transform, test_transform


class Cifar10MetaDataset(Cifar10Dataset):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                 tasks: List[int],
                 kway: int,
                 download: bool = True) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
        """
        train_transform, test_transform = self.get_transforms()

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=download,
            transform=test_transform)
        self.split_dataset(tasks, False)

        self.cls_ind = {}
        for c in range(len(tasks)):
            self.cls_ind[c] = torch.from_numpy(np.arange(c*5000, (c+1)*5000))
        self.kway = kway

    def fetch_data_loaders(self, bs, workers=4, shuf=True) -> DataLoader:
        """
        Get the Dataloader for the entire dataset
        Args:
            - shuf       : Shuffle
            - wtd_loss   : Dataloader also has wts along with targets
            - wtd_sampler: Sample data from dataloader with weights
                         according to self.tr_wts
        """
        loaders = []

        loaders.append(DataLoader(
            self.trainset, num_workers=workers, pin_memory=True,
            worker_init_fn=wif, batch_sampler=EpisodicSampler(
                self.cls_ind, self.kway, bs)))

        loaders.append(DataLoader(
            self.trainset, batch_size=200, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=wif))

        loaders.append(DataLoader(
            self.testset, batch_size=200, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=wif))

        return loaders


class Cifar100MetaDataset(Cifar100Dataset):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                 tasks: List[int],
                 kway: int,
                 download: bool = True) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
        """
        train_transform, test_transform = self.get_transforms()

        self.trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=download,
            transform=test_transform)
        self.split_dataset(tasks, False)

        self.cls_ind = {}
        for c in range(len(tasks)):
            self.cls_ind[c] = torch.from_numpy(np.arange(c*500, (c+1)*500))
        self.kway = kway

    def fetch_data_loaders(self, bs, workers=4, shuf=True) -> DataLoader:
        """
        Get the Dataloader for the entire dataset
        Args:
            - shuf       : Shuffle
            - wtd_loss   : Dataloader also has wts along with targets
            - wtd_sampler: Sample data from dataloader with weights
                         according to self.tr_wts
        """
        loaders = []

        loaders.append(DataLoader(
            self.trainset, num_workers=workers, pin_memory=True,
            worker_init_fn=wif, batch_sampler=EpisodicSampler(
                self.cls_ind, self.kway, bs)))

        loaders.append(DataLoader(
            self.trainset, batch_size=200, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=wif))

        loaders.append(DataLoader(
            self.testset, batch_size=200, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=wif))

        return loaders


class EpisodicSampler(Sampler):
    """
    Clever implementation from https://github.com/jsalbert/prototypical-networks/blob/master/samplers/episodic_batch_sampler.py
    Doesn't assume all classes have the same number of samples
    Does randomly sampling so not all samples used in every epoch
    """
    def __init__(self, lab_ind, k_way, n_samples):
        '''
        Sampler that yields batches per n_episodes without replacement.
        Batch format: (c_i_1, c_j_1, ..., c_n_way_1, c_i_2, c_j_2, ... , c_n_way_2, ..., c_n_way_n_samples)
        Args:
            lab_ind: Indices for each label
            k_way: Number of classes to sample
            k_samples: Number of samples per episode (Usually n_query + n_support)
        '''
        self.lab_ind = lab_ind
        self.k_way = k_way
        self.n_samples = n_samples
        self.n_cls = len(lab_ind)

        num_groups = 0
        for lab in lab_ind:
            num_groups += len(lab_ind[lab]) // n_samples

        self.n_episodes = num_groups // k_way

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(self.n_cls)[:self.k_way]
            for c in classes:
                l = self.lab_ind[int(c)]
                pos = torch.randperm(len(l))[:self.n_samples]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


if __name__ == "__main__":
    dataset = Cifar10MetaDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    loaders = dataset.fetch_data_loaders(16, 4)
    for x in loaders[0]:
        break
    import ipdb; ipdb.set_trace()

