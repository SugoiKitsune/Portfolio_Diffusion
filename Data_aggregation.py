# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:04:55 2023

@author: Andrey
"""

import pandas as pd
import numpy as np
import torch
import datetime
import time
import dateutil
from gensim.models import word2vec
import gensim.downloader as api
data_path = r'C:\\Personal\Personal documents\Studies and courses\CS 236 - Deep Generative Models\Project\Data'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate
word2vec_model = api.load("word2vec-google-news-300")

def pull_funds_data():
    
    # Pull list of ETF and funds 
    #At the moment it is only one date 01/09/2023
    Funds_folder = r'C:\Personal\Personal documents\Studies and courses\CS236 - Deep Generative Models\Project\Data\Composition_agg.xlsm'
    funds_db = pd.read_excel(Funds_folder, sheet_name='Combined')
    #funds_list = funds_db[]
    #Also pull all US stocks from downloaded stocks data through Tiingo API
    

def preprocess_data():
    selected_date = "2023-09-01"
    
    # Aggregate stocks information from american exchanges   
    prices_NASDAQ = pd.read_pickle(data_path+"\\prices_NASDAQ.pickle")
    prices_NYSE = pd.read_pickle(data_path+"\\prices_NYSE.pickle")
    #prices_NASDAQ['date'] = prices_NASDAQ['date'].apply(lambda x: dateutil.parser.isoparse(x).date())
    prices_NASDAQ['date'] = pd.to_datetime(prices_NASDAQ['date'].apply(pd.Timestamp))
    prices_NYSE['date'] = pd.to_datetime(prices_NYSE['date'].apply(pd.Timestamp))

    prices_combined = prices_NASDAQ.append(prices_NYSE)
    #prices_combined['date'] = prices_combined['date'].apply(lambda x: dateutil.parser.isoparse(x).date())
    prices_combined['date'] = pd.to_datetime(prices_combined['date'], format='%d/%m/%Y').dt.date
    
    # Save Aggregated daily prices for the selected tickers
    selected_date=datetime.date(2023, 9, 1)
    daily_prices = prices_combined[prices_combined['date'] == selected_date]
    daily_prices.to_csv(data_path + "\\daily_asset_prices.csv", index=False)

    tickers_combined = prices_combined['ticker'].drop_duplicates()
    tickers_combined.to_csv(data_path+"\\combined_tickers.csv")

    # Aggregate fundamentals information from the exchanges
    fundamentals_NASDAQ = pd.read_pickle(data_path + "\\fundamentals_df_new_NASDAQ.pickle")
    fundamentals_NYSE = pd.read_pickle(data_path + "\\fundamentals_NYSE.pickle")
    fundamentals_combined = fundamentals_NASDAQ.append(fundamentals_NYSE)
    fundamentals_combined['date'] = fundamentals_combined['date'].apply(lambda x: dateutil.parser.isoparse(x).date())
    #fundamentals_combined['date'] = pd.to_datetime(fundamentals_combined['date'], format='%d/%m/%Y').dt.date
    daily_fundamentals = fundamentals_combined[fundamentals_combined['date'] == selected_date]
    daily_fundamentals.to_csv(data_path + "\\daily_fundamentals.csv", index=False)

    
def text_to_vectors(text):
    # Load a pre-trained Word2Vec model (you can also train your own)
  
    vectors = [word2vec_model[word] for word in text if word in word2vec_model.key_to_index]
    if vectors:
        return sum(vectors) / len(vectors)  # Average the word vectors
    else:
        return np.zeros_like(word2vec_model['example'])

def data_combination():
    # Open the stock details 
    stocks_df = pd.read_excel(data_path+"\\tickers_format.xlsx", sheet_name='assets_reworked')    
    stocks_list = stocks_df['BBG Ticker'].drop_duplicates().tolist()
    
    # Open the fund composition file and filter on funds with available portfolio compositions
    funds_folder = data_path+'\\Composition_agg.xlsm'
    funds_composition_db = pd.read_excel(funds_folder, sheet_name='Combined')
    funds_composition_db = funds_composition_db.rename(columns={"id.ORIG_IDS": "Fund ticker", "id.WEIGHTS": "Weights", "id": "Assets ticker"})
    
    asset_in_composition_list = funds_composition_db["Assets ticker"].drop_duplicates().tolist()
    available_stocks_list  = [elem for elem in stocks_list if elem in asset_in_composition_list]
    # Create a DataFrame with the list of assets
    all_assets_df = pd.DataFrame({'Asset': available_stocks_list})
    
    # Handle duplicates by summing weights
    funds_composition_db_agg = funds_composition_db.groupby(['Fund ticker', 'Assets ticker']).agg({'Weights': 'sum'}).reset_index()
    
    # Create a pivot table from funds_composition_db
    pivot_table = pd.pivot_table(funds_composition_db_agg, 
                             index='Assets ticker', 
                             columns='Fund ticker', 
                             values='Weights', 
                             aggfunc=lambda x: x[0] if isinstance(x, list) else x)
    
    # Fill NaN values with 0
    pivot_table.fillna(0, inplace=True)
    
    result_df = pd.merge(all_assets_df, pivot_table, left_on='Asset', right_index=True, how='left')
    
    # Fill NaN values with 0 in the merged DataFrame
    result_df.fillna(0, inplace=True)

    # Normalize each column so that weights sum to 1 for every fund
    result_df.iloc[:, 1:] = result_df.iloc[:, 1:].div(result_df.iloc[:, 1:].sum(axis=0), axis=1)
    
    #result_df.to_csv(data_path + "\\composition_df.csv", index=False)
    #result_df represents normalized weights for every portfolio and each column should be the output of the generative model  

    #Now let's convert funds descriptive data into input matrix
    fund_list  = funds_composition_db["Fund ticker"].drop_duplicates().tolist()

    # Open the funds descriptive database
    funds_desc_db = pd.read_excel(data_path+"\\Funds_details.xlsx", sheet_name='Fund_data_cleaned')    
    #funds_desc_db = funds_desc_db.rename(columns={"id.ORIG_IDS": "Fund ticker"})
    
    filtered_funds_desc_db = funds_desc_db[funds_desc_db['Ticker'].isin(fund_list)]
    
    # Extract the unique column names from 'Ticker' column in filtered_funds_db
    filtered_funds_column_names = filtered_funds_desc_db['Ticker'].unique()

    # Extract the column names from results_df
    result_column_names = result_df.columns
    extra_column_names = set(result_column_names) - set(filtered_funds_column_names)

    result_df = result_df.drop(columns=extra_column_names, errors='ignore')
    result_df = result_df.set_index(all_assets_df['Asset'])
    funds_set = set(filtered_funds_column_names)
    # Preprocess and tokenize the text data in a specific column (e.g., 'Fund Objective')
    #text_data = funds_desc_db['Fund Objective'].str.lower().str.split()
    
    # Preprocess the 'Fund Objective' column
    #funds_desc_db['Fund Objective'] = funds_desc_db['Fund Objective'].fillna('').astype(str)

    # Tokenize the text data
    #funds_desc_db['Fund Objective'] = funds_desc_db['Fund Objective'].str.lower().str.split()

    # List of columns you want to convert to word vectors
    
    columns_to_convert = ['Manager', 'Type', 'Fund Asset Class Focus', 'Fund Geographical Focus', 'Fund Industry Focus', 'Fund Strategy', 'Fund Market Cap Focus']
    
    # Create a new DataFrame for storing vectorized columns
    vectorized_df = pd.DataFrame()

    # Preprocess and tokenize text data in the selected columns
    for column in columns_to_convert:
        filtered_funds_desc_db[column] = filtered_funds_desc_db[column].fillna('').astype(str)
        filtered_funds_desc_db[column] = filtered_funds_desc_db[column].str.lower().str.split()
            
    for column in columns_to_convert:
        filtered_funds_desc_db[column + ' Vector'] = filtered_funds_desc_db[column].apply(text_to_vectors)
        vectorized_df[column + ' Vector'] = filtered_funds_desc_db[column + ' Vector']
    # Apply the function to create a new column with word vectors
    #funds_desc_db['Fund Objective Vector'] = funds_desc_db['Fund Objective'].apply(text_to_vectors)

    # Create a list to store matrices (each matrix is a DataFrame)
    matrix_list = []

    # Iterate over rows and convert each vectorized column into a matrix
    for _, row in vectorized_df.iterrows():
        matrix_row = {}
        for col in columns_to_convert:
            # Check if the result is iterable before extending the matrix_row
            vectors = row[col + ' Vector']
            if vectors is not None and isinstance(vectors, (list, np.ndarray)):
                matrix_row[col] = vectors
    
        # Convert the matrix_row to a DataFrame and append it to the list
        matrix_df = pd.DataFrame(matrix_row)
        matrix_list.append(matrix_df)

    # Display the resulting list of matrices (DataFrames)

    # Sample one row from the DataFrame
    sampled_row = matrix_df.sample(1).to_numpy()
    
    # Reshape the sampled row to the original 3D structure (300x7)
    #reshaped_sample = sampled_row.reshape((7, 300))

    return matrix_list, result_df, funds_set


def data_prep_nn(matrix_list, result_df):
    # Convert matrix_list (list of DataFrames) to a list of NumPy arrays
    matrix_list_numpy = [df.to_numpy(dtype=np.float32) for df in matrix_list]

    # Convert matrixList and funds_composition to PyTorch tensors
    # Assuming each DataFrame has shape 300x7
    matrix_list_numpy = [df.to_numpy(dtype=np.float32) for df in matrix_list]

    # Convert the list of NumPy arrays to a PyTorch tensor
    matrix_list_tensor = torch.tensor(matrix_list_numpy, dtype=torch.float32)
    
    # Assuming matrix_list_tensor is your input data and results_df contains the asset weights
    input_data = matrix_list_tensor.view(-1, 300 * 7)  # Reshape data if needed
    #target_data = pd.DataFrame(target_data, columns=result_df['Fund_Name'])
    target_data = result_df.T.to_numpy()  # Convert results_df to a NumPy array
    
    target_data_df = pd.DataFrame(target_data)
  
    target_data_df = pd.DataFrame(target_data, columns=result_df.index)    
    output_data_named = target_data_df.set_index(result_df.columns)
    output_data = np.nan_to_num(target_data)
    # Ensure the number of samples matches between input_data and target_data
    assert input_data.shape[0] == output_data.shape[0], "Number of samples mismatch between input_data and target_data"

    # Split the data into training and validation sets
    input_data_train, input_data_val, output_data_train, output_data_val = train_test_split(
        input_data, output_data, test_size=0.2, random_state=42)
     
    # Check if any value is non-zero along rows (axis=0)
    non_zero_columns = np.any(output_data != 0, axis=0)

    # Filter out columns where all values are zero
    filtered_output_data = output_data[:, non_zero_columns]
     
    # Filter out columns named where all values are zero
    filtered_output_data_named = output_data_named.loc[:, non_zero_columns]
    
    
    return input_data, filtered_output_data, filtered_output_data_named




def main_script():
    matrix_list, result_df, funds_set = data_combination()
    input_data, output_data, output_data_named  = data_prep_nn(matrix_list, result_df)
    
    #input_data.to_pickle(data_path+"\\input_data_context.pickle")
    torch.save(input_data, data_path+"\\input_data_context.pickle")
    torch.save(output_data, data_path+"\\output_data.pickle")
    output_data_named.to_pickle(data_path+"\\output_data_named.pickle")
    
    return input_data, output_data, output_data_named








