# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 06:24:41 2023

@author: Andrey
"""

"""TVAE module."""

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, Softmax
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#from ctgan.data_transformer import DataTransformer
#from ctgan.synthesizers.base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        layers = []
        # Create layers based on decompress_dims
        for item in decompress_dims:
            layers.append(Linear(dim, item))
            dim = item

        # Additional layers
        layers.append(Linear(dim, dim * 2))
        layers.append(Linear(dim * 2, data_dim))

        # Output layer
        layers.append(Linear(data_dim, data_dim))
        self.seq = Sequential(*layers)
        self.softmax = Softmax(dim=-1)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)
        
    def forward(self, input_):
        """Decode the passed `input_`."""
        output = self.seq(input_)
        return output, self.sigma 


def _loss_function(recon_x, x, sigmas, mu, logvar, factor):
    st = 0
    loss = []
    rec_loss = 0
    #print(x.size(1))
    for i in range(x.size(1)):
        std = sigmas[st]
        # print(std)
        #print(st)
        #print(x[:, st])
        eq = x[:, st] - recon_x[:, st]
        rec_loss += torch.sum(torch.abs(eq))
        loss.append((eq ** 2 / 2 / (std ** 2)).sum())
        #loss.append(torch.log(std) * x.size()[0])
        st += 1
        
    #recons_loss = mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return rec_loss * factor / x.size()[0], KLD/x.shape[0]

class TVAE:
    """TVAE."""

    def __init__(
        self,
        embedding_dim=1024,
        compress_dims=(1024, 1024),
        decompress_dims=(1024, 1024),
        l2scale=1e-5,
        batch_size=64,
        epochs=80,
        loss_factor=25,
        cuda=False,
        verbose=False,
        learning_rate=0.001, gradient_clip_value=5.0,
        data_dim = 5341
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.gradient_clip_value = gradient_clip_value
        
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = 'cuda'

        self._device = torch.device(device)
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
    
    def initialize_models(self, data_dim):
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
    
    def save_model(self, encoder_path='encoder_model.pth', decoder_path='decoder_model.pth'):
        if self.encoder is not None and self.decoder is not None:
            torch.save(self.encoder.state_dict(), encoder_path)
            torch.save(self.decoder.state_dict(), decoder_path)
        else:
            print("Error: Encoder or Decoder is not initialized.")


    def load_model(self, encoder_path='encoder_model.pth', decoder_path='decoder_model.pth'):
        if self.encoder is not None and self.decoder is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print("Error: Encoder or Decoder is not initialized.")
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        #self.transformer = DataTransformer()
        #self.transformer.fit(train_data, discrete_columns)
        #train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=False)

        data_dim = train_data.shape[1]
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        self.optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2scale)

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Epoch: {epoch}, Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(epoch=0, loss=0))
        accumulation_steps = 4 
        best_loss = float('inf')  # Initialize with positive infinity
        for epoch in iterator:
            loss_values = []
            batch = []
            print('Starting processing epoch number '+ str(epoch))
            for id_, data in enumerate(loader):
                #print("Encoder Device:", next(self.encoder.parameters()).device)
                #print("Decoder Device:", next(self.decoder.parameters()).device)
                self.optimizerAE.zero_grad()
                #print(data.shape)
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                #print('Starting loss computation..')
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar, self.loss_factor)
                #print(f'Rec loss is: {loss_1}')
                #print(f'KL divergence is: {loss_2}')
                loss = (loss_1 + loss_2)
                
                if loss.detach().cpu().item() < best_loss:
                    best_loss = loss.detach().cpu().item()
                    # Save the best state (encoder and decoder weights)
                    self.save_model(encoder_path='best_encoder_model.pth', decoder_path='best_decoder_model.pth')
    
                #loss = torch.clamp(loss, min=-1e6, max=1e6)  
                loss.backward()
                if (id_ + 1) % accumulation_steps == 0:
                    self.optimizerAE.step()
                    self.optimizerAE.zero_grad()
                #self.decoder.sigma.data.clamp_(0.00, 1.0)

                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())

            epoch_loss_df = pd.DataFrame({
                'Epoch': [epoch] * len(batch),
                'Batch': batch,
                'Loss': loss_values
            })
            # Calculate the average losses at the end of the epoch
            average_rec_loss = np.mean(loss_values)
            print(f'Average Rec loss at Epoch {epoch}: {average_rec_loss:.3f}')
    
            # Optionally, you can also print the average KL divergence
            average_kl_divergence = np.mean(loss_2.detach().cpu().item())
            print(f'Average KL Divergence at Epoch {epoch}: {average_kl_divergence:.3f}')
            print(f'Score at Epoch {epoch}: {loss_values[-1]:.3f}')
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(
                        loss=loss.detach().cpu().item()))
           
            training_function_path = 'VAE_training.xlsx'  # Provide the desired file path
            epoch_loss_df.to_excel(training_function_path, index=False)
            
            
    def sample(self, samples):
        """Sample data similar to the training data.
        Args:
            samples (int):
                Number of rows to sample.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            #fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return data

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
        
        
def generated_sample(data_vector):
    # Create an instance of TVAE
    embedding_dim = 2048
    compress_dims = (2048, 2048)
    decompress_dims = (2048, 2048)

    #Specify parameters of pre-saved TVAE
    loaded_tvae = TVAE(embedding_dim, compress_dims, decompress_dims)
        
    #Load the pre-trained model with map_location=torch.device('cpu')
    encoder_state = torch.load('encoder_model.pth', map_location=torch.device('cpu'))
    decoder_state = torch.load('decoder_model.pth', map_location=torch.device('cpu'))
    # Set device to CUDA if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_tvae.set_device(device)
    data_vector = torch.tensor(data_vector, dtype=torch.float32).to(loaded_tvae._device)
   
    data_vector = data_vector.to(device)
    # Check the input data_vector
    #print("Input Data Vector:", data_vector)
    # Print information about the device
    #print("Model Device:", loaded_tvae._device)
    # print("Input Data Vector Device:", data_vector.device)

    # Switch the encoder and decoder to evaluation mode
    loaded_tvae.encoder.train(False)
    loaded_tvae.decoder.train(False)
    
    
    # Pass data_vector through the encoder
    mu, std, logvar = loaded_tvae.encoder(data_vector)

    # Assuming you want to generate a sample from the decoder
    eps = torch.randn_like(std)
    emb = eps * std + mu

    # Pass emb through the decoder
    reconstructed_data, sigmas = loaded_tvae.decoder(emb)

    # Convert reconstructed_data to numpy
    reconstructed_data_np = reconstructed_data.detach().cpu().numpy()

    # Print reconstructed data for debugging
    print("Reconstructed Data:", reconstructed_data_np)
    return reconstructed_data_np

def VAE_check():
    data_vector1 = output_data[1]  # Replace with your data vector
    data_vector2 = output_data[2]  # Replace with another data vector
    generated_sample_result1 = generated_sample(data_vector1)
    generated_sample_result2 = generated_sample(data_vector2)
    generated_sample_result1 == generated_sample_result2
    
    #Convert NumPy arrays to PyTorch tensors
    data_vector1 = torch.tensor(data_vector1, dtype=torch.float32)
    data_vector2 = torch.tensor(data_vector2, dtype=torch.float32)
    generated_sample_result1 = torch.tensor(generated_sample_result1, dtype=torch.float32)
    generated_sample_result2 = torch.tensor(generated_sample_result2, dtype=torch.float32)
    
    absolute_difference_1 = torch.abs(generated_sample_result1 - data_vector1)
    absolute_difference_2 = torch.abs(generated_sample_result2 - data_vector2)

    # Convert the PyTorch tensor to a NumPy array
    absolute_difference_np_1 = absolute_difference_1.numpy()
    absolute_difference_np_2 = absolute_difference_2.numpy()
    
    # Check if the absolute differences are below a certain threshold, e.g., 1e-6
    threshold = 1e-6
    are_equal_1 = torch.all(absolute_difference_1 < threshold)
    are_equal_2 = torch.all(absolute_difference_2 < threshold)

    # Create a DataFrame with the data
    combined_check = pd.DataFrame({
        'Data Vector 1': data_vector1.numpy(),
        'Generated Sample 1': generated_sample_result1.numpy(),
        'Absolute Difference 1': absolute_difference_np_1,
        'Data Vector 2': data_vector2.numpy(),
        'Generated Sample 2': generated_sample_result2.numpy(),
        'Absolute Difference 2': absolute_difference_np_2
    })
    
    # Save the DataFrame to an Excel file
    excel_file_path = 'VAE_sample_output.xlsx'  # Provide the desired file path
    combined_check.to_excel(excel_file_path, index=False)
    
    return combined_check