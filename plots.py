# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:20:32 2023

@author: Andrey
"""

import gc
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def sample_batch(size, output_data, input_data):
    # Assuming output_data has columns representing asset weights and rows representing samples
    num_samples  = output_data.shape[0]
    # Choose a random subset of samples (rows)
    selected_sample_indices = np.random.choice(num_samples, size=size, replace=False)
    # Select the corresponding asset weights for the selected samples
    batch_data = output_data[selected_sample_indices, :]
    batch_context_data = input_data[selected_sample_indices, :]
    # You can perform any additional processing or scaling here if needed
    # For example, normalizing the data:
    #batch_data = batch_data / np.max(batch_data, axis=0)
    return batch_data, batch_context_data


def plot(model, output_data, input_data):
    plt.figure(figsize=(10, 6))
    x0 = sample_batch(100, output_data, input_data)
    x20 = model.forward_process(torch.from_numpy(x0).to(device), 20)[-1].data.cpu().numpy()
    x40 = model.forward_process(torch.from_numpy(x0).to(device), 40)[-1].data.cpu().numpy()
    data = [x0, x20, x40]
    for i, t in enumerate([0, 20, 39]):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0: plt.title(r'$t=0$', fontsize=17)
        if i == 1: plt.title(r'$t=\frac{T}{2}$', fontsize=17)
        if i == 2: plt.title(r'$t=T$', fontsize=17)

    samples = model.sample(100, device)
    for i, t in enumerate([0, 20, 40]):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(samples[40 - t][:, 0].data.cpu().numpy(), samples[40 - t][:, 1].data.cpu().numpy(),
                    alpha=.1, s=1, c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
    plt.savefig(f"Imgs/diffusion_model.png", bbox_inches='tight')
    plt.close()




def plot_filtered_histogram(model, output_data, threshold=0.01, bins=50):
    plt.figure(figsize=(12, 6))

    # Sample original portfolio weights
    x0 = sample_batch(100, output_data, input_data)

    # Identify assets with weights greater than 1%
    filtered_assets = np.where(np.sum(x0, axis=0) > threshold)[0]
    filtered_x0 = x0[:, filtered_assets]

    # Plot column chart for the filtered assets
    plt.subplot(2, 3, 1)
    plt.bar(range(filtered_x0.shape[1]), np.mean(filtered_x0, axis=0))
    plt.title(f'Mean Weights of Filtered Assets\n(> {threshold * 100}%)')
    plt.xlabel('Asset Index')
    plt.ylabel('Mean Weight')

    # Forward process at different time steps
    x20 = model.forward_process(torch.from_numpy(filtered_x0).to(device), 20)[-1].data.cpu().numpy()
    x40 = model.forward_process(torch.from_numpy(filtered_x0).to(device), 40)[-1].data.cpu().numpy()

    data = [filtered_x0, x20, x40]

    for i, t in enumerate([0, 20, 39]):
        plt.subplot(2, 3, 2 + i)
        plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0: plt.title(r'$t=0$', fontsize=17)
        if i == 1: plt.title(r'$t=\frac{T}{2}$', fontsize=17)
        if i == 2: plt.title(r'$t=T$', fontsize=17)

    # Sample from the model at different time steps
    samples = model.sample(100, device)

    for i, t in enumerate([0, 20, 40]):
        plt.subplot(2, 3, 5 + i)
        plt.scatter(samples[40 - t][:, 0].data.cpu().numpy(), samples[40 - t][:, 1].data.cpu().numpy(),
                    alpha=.1, s=1, c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)

    plt.tight_layout()
    plt.savefig(f"Imgs/diffusion_model_filtered_column_chart.png", bbox_inches='tight')
    plt.show("Imgs/diffusion_model_filtered_histogram.png", bbox_inches='tight')
    plt.close()


def plot_single_asset(model, output_data, output_data_named, input_data, asset_index=0, threshold=0.01):
    device = 'cpu'
    plt.figure(figsize=(15, 9))
    # Sample original portfolio weights
    x0, y0 = sample_batch(100, output_data, input_data)

    # Use only one fund (asset) for simplicity
    input_context = y0[asset_index, :].unsqueeze(0)
    single_asset = x0[asset_index, :]
    #single_asset = np.where(single_asset)
    single_asset_named = pd.Series(single_asset, index=output_data_named.columns.str.replace(" US Equity", ""))
    
    # Filter assets with original weights greater than the threshold
    relevant_indices_named = single_asset_named[single_asset_named > threshold].index
    relevant_indices = np.where(single_asset > threshold)[0]
    # Plot column chart for the selected asset
    #plt.subplot(2, 3, 1)
    #plt.bar(range(1, len(single_asset_weights) + 1), single_asset_weights)
    #plt.title(f'Original Weights of Asset {asset_index + 1}')
    #plt.xlabel('Sample Index')
    #plt.ylabel('Weight')

    # Forward process at different time steps
    x20 = model.forward_process(torch.from_numpy(single_asset).to(device), 20)[-1].data.cpu().numpy()
    x40 = model.forward_process(torch.from_numpy(single_asset).to(device), 40)[-1].data.cpu().numpy()

    # Plot only for relevant indices
    plt.figure(figsize=(20, 10))
    data_full = [single_asset, x20, x40]
    data = [single_asset[relevant_indices], x20[relevant_indices], x40[relevant_indices]]
    for i, t in enumerate([0, 20, 40]):
        plt.subplot(2, 3, 1 + i)
        plt.bar(relevant_indices_named, data[i], width=0.8, align='center')
        plt.title(f'Time Step: {t}')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.xticks(rotation='vertical') 
    
    denoised_samples = []        
    generated_samples = []
    
    generated_samples = model.sample(model.n_steps, input_context, device)
    # Convert the list of PyTorch tensors to a combined NumPy array
    combined_samples_np = np.concatenate([tensor.detach().cpu().numpy() for tensor in generated_samples], axis=0)

    # Filter the combined NumPy array on relevant indices
    filtered_samples_np = combined_samples_np[:, relevant_indices]

    # Plot specific time steps for relevant indices
    plt.figure(figsize=(20, 10))
    for i, t in enumerate([0, 20, 40]):
        if t < len(filtered_samples_np):
            plt.subplot(2, 3, i+1)
            plt.bar(relevant_indices_named, filtered_samples_np[t], width=0.8, align='center')
            plt.title(f'Generated Sample at Time Step {t}')
            plt.xlabel('Assets')
            plt.ylabel('Weight')
            plt.xticks(rotation='vertical')

    plt.show()

def plot_individual_text_sample(model, output_data, output_data_named, input_data, asset_index=0, threshold=0.01):
    
    # Assuming you have df_output_data and df_output_data_named dataframes

    # Filter df_output_data_named
    # Select the text index
    selected_text_index = 'BUFFT US Equity'

    # Reindex df_output_data_named
    output_data_named_reindexed = output_data_named.reset_index()

    # Find the corresponding numeric index
    numeric_index = output_data_named_reindexed[output_data_named_reindexed['index'] == selected_text_index].index.to_numpy()[0]
    
    # Filter df_output_data using the numeric index
    filtered_output_named_sample = output_data_named[output_data_named.index == selected_text_index]
    filtered_output_sample = output_data[numeric_index]
    filtered_input_sample = input_data[numeric_index]
    input_context = filtered_input_sample 
    
    single_asset = filtered_output_sample
    single_asset_named = pd.Series(single_asset, index=filtered_output_named_sample.columns.str.replace(" US Equity", ""))
    
    # Filter assets with original weights greater than the threshold
    relevant_indices_named = single_asset_named[single_asset_named > threshold].index
    relevant_indices = np.where(single_asset > threshold)[0]

    # Forward process at different time steps
    x20 = model.forward_process(torch.from_numpy(single_asset).to(device), 20)[-1].data.cpu().numpy()
    x40 = model.forward_process(torch.from_numpy(single_asset).to(device), 40)[-1].data.cpu().numpy()
    x60 = model.forward_process(torch.from_numpy(single_asset).to(device), 60)[-1].data.cpu().numpy()
    
    # Plot only for relevant indices
    plt.figure(figsize=(20, 10))
    data_full = [single_asset, x20, x40, x60]
    data = [single_asset[relevant_indices], x20[relevant_indices], x40[relevant_indices], x60[relevant_indices]]
    for i, t in enumerate([0, 20, 40, 60]):
        plt.subplot(2, 3, 1 + i)
        plt.bar(relevant_indices_named, data[i], width=0.8, align='center')
        plt.title(f'Time Step: {t}')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.xticks(rotation='vertical') 
    
    denoised_samples = []        
    generated_samples = []
    
    generated_samples = model.sample(model.n_steps, input_context, device)
    # Convert the list of PyTorch tensors to a combined NumPy array
    combined_samples_np = np.concatenate([tensor.detach().cpu().numpy() for tensor in generated_samples], axis=0)

    # Filter the combined NumPy array on relevant indices
    filtered_samples_np = combined_samples_np[:, relevant_indices]

    # Plot specific time steps for relevant indices
    plt.figure(figsize=(20, 10))
    for i, t in enumerate([0, 20, 40, 60]):
        if t < len(filtered_samples_np):
            plt.subplot(2, 3, i+1)
            plt.bar(relevant_indices_named, filtered_samples_np[t], width=0.8, align='center')
            plt.title(f'Generated Sample at Time Step {t}')
            plt.xlabel('Assets')
            plt.ylabel('Weight')
            plt.xticks(rotation='vertical')

    plt.show()
    
    return

def plot_samples(model, device, output_data, input_data, t, sample_size=5000):
    input_tensor = torch.from_numpy(input_data).float().to(device).unsqueeze(0)  # Add batch dimension

    # Forward process to get the initial sample
    with torch.no_grad():
        mu_posterior, sigma_posterior, xt = model.forward_process(input_tensor, t)

    # Generate samples using the reverse process
    samples = [xt.squeeze().cpu().numpy()]  # Start with the initial sample
    for _ in range(model.n_steps - t):
        _, _, x = model.reverse(samples[-1].reshape(1, -1), t + len(samples))
        samples.append(x.squeeze().cpu().numpy())

    # Convert to NumPy array
    samples = np.array(samples)

    plt.figure(figsize=(10, 6))
    for i, step in enumerate(range(t, model.n_steps + 1)):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(samples[step - t][:, 0], samples[step - t][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if step == t:
            plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0:
            plt.title(r'$t=0$', fontsize=17)
        else:
            plt.title(r'$t={} (t+{})$'.format(step, step - t), fontsize=17)

    samples = model.sample(sample_size, device)
    for i, step in enumerate(range(model.n_steps, t, -1)):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(samples[model.n_steps - step][:, 0].data.cpu().numpy(),
                    samples[model.n_steps - step][:, 1].data.cpu().numpy(),
                    alpha=.1, s=1, c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if step == t:
            plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        plt.title(r'$t={} (t+{})$'.format(step, model.n_steps - step), fontsize=17)

    plt.show()


def plot_samples(model, device, output_data, t, sample_size=100):
    # Sample input data from the output data
    input_data = sample_batch(sample_size, output_data)

    input_tensor = torch.from_numpy(input_data).float().to(device)

    # Forward process to get the initial sample
    with torch.no_grad():
        mu_posterior, sigma_posterior, xt = model.forward_process(input_tensor, t)

    # Generate samples using the reverse process
    samples = [xt.squeeze().cpu().numpy()]  # Start with the initial sample
    for _ in range(model.n_steps - t):
        _, _, x = model.reverse(samples[-1].reshape(1, -1), t + len(samples))
        samples.append(x.squeeze().cpu().numpy())

    # Convert to NumPy array
    samples = np.array(samples)

    plt.figure(figsize=(10, 6))
    for i, step in enumerate(range(t, model.n_steps + 1)):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(samples[step - t][:, 0], samples[step - t][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if step == t:
            plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0:
            plt.title(r'$t=0$', fontsize=17)
        else:
            plt.title(r'$t={} (t+{})$'.format(step, step - t), fontsize=17)

    samples = model.sample(sample_size, device)
    for i, step in enumerate(range(model.n_steps, t, -1)):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(samples[model.n_steps - step][:, 0].data.cpu().numpy(),
                    samples[model.n_steps - step][:, 1].data.cpu().numpy(),
                    alpha=.1, s=1, c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if step == t:
            plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        plt.title(r'$t={} (t+{})$'.format(step, model.n_steps - step), fontsize=17)

    plt.show()