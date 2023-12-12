# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:45:42 2023

@author: Andrey
"""
import gc
import pandas as pd
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from modules import MLP, SimpleDiffusion, GaussianDiffusion
from plots import plot, plot_single_asset
from tvae import Encoder, Decoder, TVAE

#from sklearn.datasets import make_swiss_roll

data_path = r'C:\Personal\Personal documents\Studies and courses\CS 236 - Deep Generative Models\Project\Data'
code_path = r'C:\Personal\Personal documents\Studies and courses\CS236 - Deep Generative Models\Project\Code\Diffusion'
#def sample_batch(size):
    #x, _ = make_swiss_roll(size)
    #return x[:, [2, 0]] / 10.0 * np.array([1, -1])

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

def save_model(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    
def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch    

def generate_sample(model, device, input_data, input_text_sample):
    # Assuming input_data is a NumPy array
    input_tensor = torch.from_numpy(input_data).float().to(device).unsqueeze(0)  # Add batch dimension
    #input_text_sample = input_text_sample).float().to(device).unsqueeze(0) 
    # Choose a random time step for generation
    t = np.random.randint(2, model.n_steps + 1)
    # Forward process to generate the output sample
    #with torch.no_grad():
        #mu_posterior, sigma_posterior, xt = model.forward_process(input_tensor, t)
        #mu, sigma, generated_sample = model.reverse(xt,input_text_sample, t)
        
    # Reverse the diffusion sequence
    for t in reversed(range(2, model.n_steps + 1)):
        with torch.no_grad():
            mu_posterior, sigma_posterior, xt = model.forward_process(input_tensor, t)
            mu, sigma, generated_sample = model.reverse(xt, input_text_sample, t)

        # Update the input tensor for the next step
        input_tensor = generated_sample

    # Convert the generated sample to NumPy array
    generated_sample = generated_sample.squeeze().cpu().numpy()
    
    return generated_sample


def train(model, optimizer, output_data, input_data, device, nb_epochs=125000, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        x0, y0 = sample_batch(batch_size, output_data, input_data)
        x0 = torch.tensor(x0, dtype=torch.float32).to(device)
        y0 = torch.tensor(y0, dtype=torch.float32).to(device)
        t = np.random.randint(2, model.n_steps + 1)
        
        
        mu_posterior, sigma_posterior, xt = model.forward_process(x0, t)
        mu, sigma, _ = model.reverse(xt, y0, t)
    
        # Print dimensions for debugging
        #print(f"sigma: {sigma.size()}, sigma_posterior: {sigma_posterior.size()}")
         # Ensure dimensions match for KL divergence calculation
        if len(sigma_posterior.size()) > 1 and len(sigma.size()) > 1:
            sigma = sigma[:, :sigma_posterior.size(1)]
        
        
        KL = (torch.log(sigma) - torch.log(sigma_posterior) + (sigma_posterior ** 2 + (mu_posterior - mu) ** 2) / (
                2 * sigma ** 2) - 0.5)
        loss = KL.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        
        # Print or log the loss every 1000 steps
        log_interval = 1000
        if epoch % log_interval == 0 and epoch > 0:
            average_loss = np.mean(training_loss[-log_interval:])
            print(f"Epoch {epoch}, Loss: {average_loss:.4f}")

        # Save the model weights every 1000 epochs
        if epoch % 40000 == 0:
            save_model(model, optimizer, epoch, f"model_checkpoint_epoch_{epoch}.pt")

def train_pipeline(model, encoder, decoder, optimizer, output_data, input_data, device, nb_epochs=50000, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        x0, y0 = sample_batch(batch_size, output_data, input_data)
        x0 = torch.tensor(x0, dtype=torch.float32).to(device)
        y0 = torch.tensor(y0, dtype=torch.float32).to(device)
        t = np.random.randint(2, model.n_steps + 1)
        
        
        # Encode the input data and generate latent variables
        mu_encoded, std_encoded, log_var_encoded = encoder(x0)
        eps = torch.randn_like(std_encoded)
        latent_var = eps * std_encoded + mu_encoded
        
        # Generate noise using the model forward process
        mu_posterior, sigma_posterior, xt = model.forward_process(latent_var, t)
        
        # Use the diffusion model to reverse the process
        mu, sigma, _ = model.reverse(xt, y0, t)
    
        # Ensure dimensions match for KL divergence calculation
        if len(sigma_posterior.size()) > 1 and len(sigma.size()) > 1:
            sigma = sigma[:, :sigma_posterior.size(1)]
        
        KL = (torch.log(sigma) - torch.log(sigma_posterior) + (sigma_posterior ** 2 + (mu_posterior - mu) ** 2) / (
                2 * sigma ** 2) - 0.5)
        loss = KL.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        
        # Print or log the loss every 100 steps
        log_interval = 100
        if epoch % log_interval == 0 and epoch > 0:
            average_loss = np.mean(training_loss[-log_interval:])
            print(f"Epoch {epoch}, Loss: {average_loss:.4f}")

        # Save the model weights every 1000 epochs
        if epoch % 40000 == 0:
            save_model(model, optimizer, epoch, f"model_checkpoint_epoch_{epoch}.pt")




def main_diffusion(mode):
    # Load your input and output data
    input_data = torch.load(data_path + "\\input_data_context.pickle")
    output_data = torch.load(data_path + "\\output_data.pickle")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the TVAE with the encoder and decoder architecture
    tvae = TVAE(embedding_dim=1024, compress_dims=(1024, 1024), decompress_dims=(1024, 1024), l2scale=1e-5,
                batch_size=200, epochs=300, loss_factor=60, cuda=device, verbose=True, learning_rate=0.001,
                gradient_clip_value=5.0, data_dim=6684)

    # Load pre-trained encoder and decoder weights
    tvae.encoder.load_state_dict(torch.load(code_path+"\pretrained_encoder.pth"))
    tvae.decoder.load_state_dict(torch.load(code_path+"\pretrained_decoder.pth"))

    # Set models to evaluation mode since you're not training them
    tvae.encoder.eval()
    tvae.decoder.eval()

    # Define your diffusion model (MLP + SimpleDiffusion)
    model_mlp = MLP(hidden_dim=64).to(device)
    diffusion_model = SimpleDiffusion(model_mlp)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)

    if mode == 'train':
        train_pipeline(diffusion_model, tvae.encoder, tvae.decoder, optimizer, output_data, input_data, device)
        save_model(diffusion_model, optimizer, 5000, "final_model.pt")

    else:
        loaded_model, loaded_optimizer, loaded_epoch = load_model(diffusion_model, optimizer, "final_model.pt")
        # Set the model to evaluation mode
        loaded_model.eval()
        diffusion_model = loaded_model

    plot(diffusion_model, output_data)

    # Generate an output sample given an input sample
    # input_sample = np.random.rand(1, 6684)  # Replace this with your actual input data
    input_sample, input_text_sample = output_data[1], input_data[1]
    
    generated_sample = generate_sample(diffusion_model, 'cuda', input_sample, input_text_sample)

    # Display or use the generated sample as needed
    print("Generated Sample:", generated_sample)



def main_simplediffusion(mode):
    input_data = torch.load(data_path+"\\input_data_context.pickle")
    output_data = torch.load(data_path+"\\output_data.pickle")
    output_data_named = torch.load(data_path+"\\output_data.pickle")
    
    
    device = 'cpu'
    model_mlp = MLP(hidden_dim=256).to(device)
    model = SimpleDiffusion(model_mlp)
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-4)
    if mode == 'train':
        train(model, optimizer, output_data, input_data, device)
        save_model(model, optimizer, 125000, "final_model_enh.pt")
    else:
        loaded_model, loaded_optimizer, loaded_epoch = load_model(model, optimizer, "final_model_simple.pt")
        # Set the model to evaluation mode
        loaded_model.eval()
        model = loaded_model
            
    plot_single_asset(model, output_data, output_data_named, input_data, asset_index=3, threshold=0.01)
    # Generate an output sample given an input sample
    #input_sample = np.random.rand(1, 6684)  # Replace this with your actual input data
    input_sample, input_text_sample = output_data[1], input_data[1]
    generated_sample = generate_sample(model, 'cpu', input_sample, input_text_sample)

    # Display or use the generated sample as needed
    print("Generated Sample:", generated_sample)


def train_gaussian_diffusion(model, optimizer, output_data, device, nb_epochs=40000, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        x0, y0 = torch.from_numpy(sample_batch(batch_size, output_data)).float().to(device)
        t = np.random.randint(2, model.timesteps + 1)
        
        # Perform forward and reverse processes
        pred_mean, pred_variance, _ = model.q_mean_variance(x_start=x0, t=t)
        pred_noise = torch.randn_like(x0)  # assuming pred_noise is standard Gaussian
        x_pred = model.q_sample(x_start=x0, t=t, noise=pred_noise)
        x_recon, _, _ = model.p_mean_variance(pred_noise=pred_noise, x=x_pred, t=t)

        # Calculate KL divergence
        KL = (
            torch.log(pred_variance) -
            torch.log(model.alphas_cumprod_prev[t - 1]) +
            (model.alphas_cumprod_prev[t - 1] ** 2 + (pred_mean - x0) ** 2) / (2 * pred_variance ** 2) - 0.5
        )
        loss = KL.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        
        # Save the model weights every 1000 epochs
        if epoch % 1000 == 0:
            save_model(model, optimizer, epoch, f"gaussian_diffusion_model_checkpoint_epoch_{epoch}.pt")
            

def main_gaussian():
    device = 'cuda'
    mlp_model = MLP(N=40, data_dim=6684, hidden_dim=2048).to(device)
    gaussian_diffusion_model = GaussianDiffusion(mlp_model, timesteps=1000).to(device)
    optimizer_gaussian_diffusion = torch.optim.Adam(gaussian_diffusion_model.parameters(), lr=1e-4)

    # Training loop for GaussianDiffusion with MLP
    train_gaussian_diffusion(gaussian_diffusion_model, optimizer_gaussian_diffusion, output_data, device)
    save_model(gaussian_diffusion_model, optimizer_gaussian_diffusion, 40000, "final_gaussian_diffusion_model.pt")
     
    loaded_model, loaded_optimizer, loaded_epoch = load_model(model, optimizer, "final_model.pt")
    # Set the model to evaluation mode
    loaded_model.eval()
    
    # Generate an output sample given an input sample
    #input_sample = np.random.rand(1, 6684)  # Replace this with your actual input data
    input_sample = output_data[2]
    generated_sample = generate_sample(loaded_model, 'cuda', input_sample)

    # Display or use the generated sample as needed
    print("Generated Sample:", generated_sample)
    