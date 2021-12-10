import xarray as xr
import fsspec
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import fsspec
import zarr
import os
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from tqdm import tqdm
import gc

mean_dataset_root = Path('input_data')
mean_ds_paths = {'spatial': mean_dataset_root / 'spatial_only_means.nc',
                 'temporal': mean_dataset_root / 'temporal_only_means.nc',
                 'spatiotemporal': mean_dataset_root / 'spatiotemporal_only_means.nc'}

def norm_ds(input_dataset: xr.Dataset, mean_dataset: xr.Dataset, feature: str):
    return (input_dataset[feature] - mean_dataset[feature + '_mean']) / mean_dataset[feature + '_std']


def get_pixel_feature_ds(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0):
    patch_half = patch_size // 2
    if access_mode == 'spatiotemporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y', 'time'])
    elif access_mode == 'temporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=x, y=y)
    elif access_mode == 'spatial':
        block = the_ds.isel(x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1), time=t)  # .reset_index(['x', 'y'])

    return block


def get_pixel_feature_vector(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0,
                             dynamic_features=None,
                             static_features=None, nan_fill=-1.0, override_whole=False):
    mean_ds_path = mean_ds_paths[access_mode]

    mean_ds = xr.open_dataset(mean_ds_path).load()
    if override_whole:
        chunk = the_ds
    else:
        chunk = get_pixel_feature_ds(the_ds, t=t, x=x, y=y, access_mode=access_mode, patch_size=patch_size, lag=lag)
    if access_mode == 'spatial':
        year = pd.DatetimeIndex([chunk['time'].values]).year[0]
    else:
        year = pd.DatetimeIndex([chunk['time'][0].values]).year[0]
    # clc
    if 'clc' in static_features:
        if year < 2012:
            chunk['clc'] = chunk['clc_2006']
        elif year < 2018:
            chunk['clc'] = chunk['clc_2012']
        else:
            chunk['clc'] = chunk['clc_2018']

    # population density
    if 'population_density' in static_features:
        chunk['population_density'] = chunk[f'population_density_{year}']
    dynamic = np.stack([norm_ds(chunk, mean_ds, feature) for feature in dynamic_features])
    if 'temp' in access_mode:
        dynamic = np.moveaxis(dynamic, 0, 1)
    static = np.stack([norm_ds(chunk, mean_ds, feature) for feature in static_features])
    dynamic = np.nan_to_num(dynamic, nan=nan_fill)
    static = np.nan_to_num(static, nan=nan_fill)
    return dynamic, static


def get_pixel_feature_ds_day(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0):
    patch_half = patch_size // 2
    if access_mode == 'spatiotemporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y', 'time'])
    elif access_mode == 'temporal':
        block = the_ds.isel(time=slice(t + 1 - lag, t + 1), x=x, y=y).reset_index(['time'])
    elif access_mode == 'spatial':
        block = the_ds.isel(x=slice(x - patch_half, x + patch_half + 1),
                            y=slice(y - patch_half, y + patch_half + 1))  # .reset_index(['x', 'y'])

    return block

def get_pixel_feature_vector_day(the_ds, t=0, x=0, y=0, access_mode='temporal', patch_size=0, lag=0,
                             dynamic_features=None,
                             static_features=None, nan_fill=-1.0, override_whole=False):
    mean_ds_path = mean_ds_paths[access_mode]
    mean_ds = xr.open_dataset(mean_ds_path).load()
    if override_whole:
        chunk = the_ds
    else:
        chunk = get_pixel_feature_ds_day(the_ds, t=t, x=x, y=y, access_mode=access_mode, patch_size=patch_size, lag=lag)

    dynamic = np.stack([norm_ds(chunk, mean_ds, feature) for feature in dynamic_features])
    if 'temp' in access_mode:
        dynamic = np.moveaxis(dynamic, 0, 1)
    static = np.stack([norm_ds(chunk, mean_ds, feature) for feature in static_features])
    dynamic = np.nan_to_num(dynamic, nan=nan_fill)
    static = np.nan_to_num(static, nan=nan_fill)
    return dynamic, static



class FireDatasetWholeDay(Dataset):
    def __init__(self, ds, day, access_mode='temporal', problem_class='classification', patch_size=0, lag=10,
                 dynamic_features=None,
                 static_features=None, nan_fill=-1.0):
        assert access_mode in ['temporal', 'spatial', 'spatiotemporal']
        assert problem_class in ['classification', 'segmentation']
        self.problem_class = problem_class
        if lag > 0:
            self.ds = ds.isel(time=range(day - lag + 1, day + 1))
        else:
            self.ds = ds.isel(time=day)
        self.override_whole = problem_class == 'segmentation'
        self.ds = self.ds.load()
        
        pixel_range = patch_size // 2
        self.pixel_range = pixel_range
        self.len_x = self.ds.dims['x']
        self.len_y = self.ds.dims['y']
        if access_mode == 'spatial':
            year = pd.DatetimeIndex([self.ds['time'].values]).year[0]
        else:
            year = pd.DatetimeIndex([self.ds['time'][0].values]).year[0]
        # clc
        if 'clc' in static_features:
            if year < 2012:
                self.ds['clc'] = self.ds['clc_2006']
            elif year < 2018:
                self.ds['clc'] = self.ds['clc_2012']
            else:
                self.ds['clc'] = self.ds['clc_2018']

        # population density
        if 'population_density' in static_features:
            self.ds['population_density'] = self.ds[f'population_density_{year}']

        if access_mode == 'spatiotemporal':
            new_ds_dims = ['time', 'y', 'x']
            new_ds_dict = {}
            for feat in dynamic_features:
                new_ds_dict[feat] = (new_ds_dims,
                                     np.pad(self.ds[feat].values,
                                            pad_width=((0, 0), (pixel_range, pixel_range), (pixel_range, pixel_range)),
                                            mode='constant', constant_values=(0, 0)))
            new_ds_dims_static = ['y', 'x']
            for feat in static_features:
                new_ds_dict[feat] = (new_ds_dims_static,
                                     np.pad(self.ds[feat].values,
                                            pad_width=((pixel_range, pixel_range), (pixel_range, pixel_range)),
                                            mode='constant', constant_values=(0, 0)))
            self.ds = xr.Dataset(new_ds_dict)

        elif access_mode == 'spatial':
            new_ds_dims = ['y', 'x']
            new_ds_dict = {}
            for feat in dynamic_features + static_features:
                new_ds_dict[feat] = (new_ds_dims,
                                     np.pad(self.ds[feat].values,
                                            pad_width=((pixel_range, pixel_range), (pixel_range, pixel_range)),
                                            mode='constant', constant_values=(0, 0)))
            new_ds_dict['time'] = self.ds['time'].values
            self.ds = xr.Dataset(new_ds_dict)

        self.patch_size = patch_size
        self.lag = lag
        self.access_mode = access_mode
        self.day = day
        self.nan_fill = nan_fill
        self.dynamic_features = dynamic_features
        self.static_features = static_features

    def __len__(self):
        if self.problem_class == 'segmentation':
            return 1
        return self.len_x * self.len_y

    def __getitem__(self, idx):
        y = idx // self.len_x + self.pixel_range
        x = idx % self.len_x + self.pixel_range
        if self.lag == 0:
            day = 0
        else:
            day = self.lag - 1
        dynamic, static = get_pixel_feature_vector_day(self.ds, day, x,
                                                   y, self.access_mode, self.patch_size,
                                                   self.lag,
                                                   self.dynamic_features,
                                                   self.static_features,
                                                   self.nan_fill, self.override_whole)
        return dynamic, static