# Public notebooks for DeepCube's Use Case 3 on Fire Hazard short-term Forecasting

This repository holds jupyter notebooks for some demonstrable results of this use case.

## Prerequisites

The notebooks are tested to run with the dependencies in the `requirements.txt` file.

To install them in your system, run `pip install -r requirements.txt`.

## Datacube Access and Plotting

The first notebook [1_UC3_Datacube_Access_and_Plotting.ipynb](1_UC3_Datacube_Access_and_Plotting.ipynb) downloads and opens the [public datacube](https://zenodo.org/record/4943354) with xarray.

Then, we demonstrate how the datacube structure allows us to easily visualize and perform analytics on our data.

## Deep Learning (DL) models inference

This second notebook [2_UC3_DL_models_inference.ipynb](2_UC3_DL_models_inference.ipynb) loads pretrained Deep Learning models that are used to create Fire Danger maps for Greece, extracting the necessary data from the datacube presented in the previous notebook. 


The methods to download the models are presented in detail in our relevant paper [available on arXiv](https://arxiv.org/abs/2111.02736), which has been accepted to the workshop on Artificial Intelligence for Humanitarian Assistance and Disaster Response at the 35th Conference on Neural Information Processing Systems (NeurIPS 2021).



