"""
DataModule for the neural network.

Loads and preprocesses the image data on demand.

@author Raúl Coterillo
@version 20-12-2022
"""

from __future__ import annotations

# torchs stuff
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv
import torch

# train-test splitting
from sklearn.model_selection import train_test_split

# arrays and all that stuff
import pandas as pd
import numpy as np

# open images
import rasterio

# system stuff
import multiprocessing as mp
from pathlib import Path

# progress bar
from tqdm import tqdm


class RunningStatistics():

    '''
    Records mean and variance of the final `n_dims` dimension over other dimensions across items. 
    
    So collecting across `(l,m,n,o)` sized items with `n_dims=1` will collect `(l,m,n)` sized statistics while with `n_dims=2` 
    the collected statistics will be of size `(l,m)`. Uses the algorithm from Chan, Golub, and LeVeque in "Algorithms for computing 
    the sample variance: analysis and recommendations": `variance = variance1 + variance2 + n/(m*(m+n)) * pow(((m/n)*t1 - t2), 2)`
    This combines the variance for 2 blocks: block 1 having `n` elements with `variance1` and a sum of `t1` and block 2 having `m` elements
    with `variance2` and a sum of `t2`. The algorithm is proven to be numerically stable but there is a reasonable loss of accuracy (~0.1% error).
    Note that collecting minimum and maximum values is reasonably inefficient, adding about 80% to the running time, and hence is disabled by default.
    '''

    # gracias random internet guy que posteó esto en un foro aleatorio, may your crops grow strong this spring
    
    def __init__(self, n_dims:int=2, record_range=False):
        """ Create attributes and initialize object. """

        self._n_dims, self._range = n_dims, record_range
        self.n, self.sum, self.min, self.max = 0, None, None, None
    
    def update(self, data: torch.Tensor):

        data = data.view(*list(data.shape[:-self._n_dims]) + [-1])
        
        with torch.no_grad():
            
            new_n, new_var, new_sum = data.shape[-1], data.var(-1), data.sum(-1)

            if self.n == 0:
            
                self.n = new_n
                self._shape = data.shape[:-1]
                self.sum = new_sum
                self._nvar = new_var.mul_(new_n)
                if self._range:
                    self.min = data.min(-1)[0]
                    self.max = data.max(-1)[0]
            
            else:

                assert data.shape[:-1] == self._shape, f"Mismatched shapes, expected {self._shape} but got {data.shape[:-1]}."

                ratio = self.n / new_n
                t = (self.sum / ratio).sub_(new_sum).pow_(2)
                
                self._nvar.add_(new_n, new_var).add_(ratio / (self.n + new_n), t)
                #self._nvar.add(new_var, alpha=new_n).add(t, alpha=ratio / (self.n + new_n))

                self.sum.add_(new_sum)
                self.n += new_n
                
                if self._range:

                    self.min = torch.min(self.min, data.min(-1)[0])
                    self.max = torch.max(self.max, data.max(-1)[0])

    @property
    def mean(self): return self.sum / self.n if self.n > 0 else None
    @property
    def var(self): return self._nvar / self.n if self.n > 0 else None
    @property
    def std(self): return self.var.sqrt() if self.n > 0 else None

    def __repr__(self):
        def _fmt_t(t: torch.Tensor):
            if t.numel() > 5: return f"tensor of ({','.join(map(str,t.shape))})"
            def __fmt_t(t: torch.Tensor):
                return '[' + ','.join([f"{v:.3g}" if v.ndim==0 else __fmt_t(v) for v in t]) + ']'
            return __fmt_t(t)
        rng_str = f", min={_fmt_t(self.min)}, max={_fmt_t(self.max)}" if self._range else ""
        return f"RunningStatistics(n={self.n}, mean={_fmt_t(self.mean)}, std={_fmt_t(self.std)}{rng_str})"

# =========================================================================== #
# =========================================================================== #

class ImageDataset(Dataset):

    def __init__(self,
            dataset: pd.DataFrame,
            image_shape: np.ndarray,
            transform = None, 
            target_transform = None
            ) -> None:

        self.dataset, self.image_shape = dataset, image_shape
        self.transform, self.target_transform = transform, target_transform

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return len(self.dataset)

    @staticmethod
    def _load_image(image_path, remove_negs=True, normalization=False):
        
        """ Load the image as a numpy array. """

        img = rasterio.open(image_path)
        concat_list = []
        
        if img.count == 5: # 1=B, 2=G, 3=R, 4=RE, 5=NIR
            channels_list = [1,2,3,4,5]
        elif img.count == 10: # 1=B, 3=G, 5=R, 7=RE, 10=NIR
            channels_list = [1,3,5,7,10]
        else:
            raise Exception("Unexpected number of channels in image %s" %image_path)
        
        for i in channels_list:
            ch_image = img.read(i)
            if remove_negs:
                ch_image[ch_image < 0] = 0.0
            concat_list.append(np.expand_dims(ch_image, -1))
        
        image = np.concatenate(concat_list, axis=-1).astype(float) # float 0-1

        if normalization:
            max_pixel_value = np.max(image, axis=2)
            max_pixel_value = np.repeat(max_pixel_value[:, :, np.newaxis], 5, axis=2)
            image = np.divide(image, max_pixel_value)
            image = image[:, :, 0:4]
            image[np.isnan(image)] = 0.0

        return image

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        row = self.dataset.iloc[idx]
        image = self._load_image(row["PLOT_FILE"])
        labels = row[["DISEASE1","DISEASE2","DISEASE3"]].values.astype(np.float32)

        # image transforms
        tr = tv.transforms.Compose([
            tv.transforms.ToTensor(),    # from numpy HxWxC to tensor CxHXW 
            tv.transforms.Resize(size=self.image_shape)
        ])

        # convert to tensort
        image = tr(image)
        labels = torch.Tensor(labels)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            olabel = self.target_transform(olabel)

        return image, labels

# =========================================================================== #
# =========================================================================== #

class ImageDataModule(LightningDataModule):

    """ DataModule for pytorch-lightning"""

    norm_cts_file = Path("norm_cts.npz")

    def __init__(self, 
            csv_file: str = "csvs/train.csv",
            test_size: float = 0.2,
            eval_size: float = 0.2,
            batch_size: int = 128,
            random_state: int = 0,
            image_shape: np.ndarray = [110,330],
            num_workers: int = mp.cpu_count()//2,
            ) -> None:
        
        super().__init__()

        dataset = pd.read_csv(csv_file)

        self.test_size = test_size
        self.eval_size = eval_size
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.random_state = random_state

        self.image_shape = image_shape

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

        idx_train_eval, idx_test = train_test_split(np.arange(len(dataset)), test_size=self.test_size, 
            random_state=self.random_state, shuffle=True)

        idx_train, idx_eval = train_test_split(idx_train_eval, test_size=self.eval_size, 
            random_state=self.random_state, shuffle=True)

        self.ds_train = ImageDataset(dataset.iloc[idx_train].reset_index(), self.image_shape)
        self.ds_eval  = ImageDataset(dataset.iloc[idx_eval].reset_index(), self.image_shape)
        self.ds_test  = ImageDataset(dataset.iloc[idx_test].reset_index(), self.image_shape)

        # calculate normalization constants (not fucking trivial)
        if not self.norm_cts_file.exists():
            stats = RunningStatistics(n_dims=2)
            print("Computing normalization constants...")
            for item in tqdm(self.ds_train):
                stats.update(item[0])
            mean, std = stats.mean, stats.std
            np.savez_compressed(self.norm_cts_file, mean=mean.numpy(), std=std.numpy())
        else:
            with np.load(self.norm_cts_file, allow_pickle=True) as data:
                mean, std = torch.from_numpy(data["mean"]), torch.from_numpy(data["std"])

        transform = tv.transforms.Compose([tv.transforms.Normalize(mean, std)])

        self.ds_train.transform = transform
        self.ds_eval.transform = transform
        self.ds_test.transform = transform

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def train_dataloader(self):
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def val_dataloader(self):
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_eval, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def test_dataloader(self):
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_dataloader(self):
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers) 


# =========================================================================== #
# # =========================================================================== #

if __name__ == "__main__":

    dm = ImageDataModule()