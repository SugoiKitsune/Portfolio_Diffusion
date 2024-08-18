![stanf_header](https://github.com/user-attachments/assets/7fe2e6a5-dc34-4545-8136-d56d7ffe5c45)


**Overview**
This project explores the application of generative modeling, specifically diffusion models, to the construction of investment portfolios. 
By leveraging techniques typically used in image generation, we aim to automate and optimize the creation of investment portfolios based on specific criteria such as asset class, geography, and strategy.
This is the Final Project for the module CS236: Deep Generative Modelling, taught by prof Stefano Ermon at Stanford University.

**Project Objectives**
Apply generative modeling techniques to financial data for portfolio generation.
Utilize diffusion models, typically used in image generation, to replicate and construct investment portfolios.
Evaluate the effectiveness of these models in generating realistic and effective portfolios based on input prompts.
Methodology
The approach is inspired by text-to-image generation models, such as the SDXL model, adapted for financial data. The core components include:
![SDXL sample](https://github.com/user-attachments/assets/cbe2f77a-62ee-4235-962d-279f0f928c6e)

**Dataset and Tokenization:**

Collected data from over 6,000 funds and ETFs focusing on US Equities.
Tokenized both descriptive and quantitative data for model training. Below is the example of the input data:

![input data sample](https://github.com/user-attachments/assets/ec142594-bc46-42cf-884a-a5b6274eb38a)


For the output model the funds and investor portfolio composition has been utilized with the precise asset allocation

![output_data_sample](https://github.com/user-attachments/assets/b5c694a1-a934-4ba2-a446-68733735c42a)

**Diffusion Model:**
VAE has been implemented to encode and decode data in a probabilistic manner, optimizing for low portfolio composition error.
Within Diffusion pipeline the classical U-net has been replaced with the multi-layer perceptron(MLP) supported by x-attention modules 
Applied to generate new portfolio samples through a forward and reverse process, gradually adding and removing noise.

![portoflio_diffusion](https://github.com/user-attachments/assets/7650b572-7e1e-4260-a27c-62968549bb76)

**Results**
The project demonstrated that diffusion models could indeed generate portfolios that, while not exact, closely replicate the structure of actual portfolios based on input prompts.

Denoising Sequence:
A reverse process showing how the model reconstructs portfolios from noisy data.
![asset_weight_denoising](https://github.com/user-attachments/assets/0056bec8-6f97-44a6-9c75-10424aef61a4)

**Example: Buffet prtfolio as of 09/2023**

An example portfolio was generated based on the prompt: "Buffett, Closed-End Fund, Thematic, Value, Large-Cap, Sep 2023". The generated portfolio closely matched Warren Buffett's actual portfolio, though with some differences in asset allocation.
![buffett](https://github.com/user-attachments/assets/99941849-13ed-4e33-be22-7f062d9a1aea)

**Future Work**

Model Accuracy: Enhancements to the VAE and diffusion models to reduce data loss and improve portfolio generation precision.

Dimensionality Expansion: Increasing model scope to handle multiple asset classes and international markets.

Alternative Architectures: Exploring models that incorporate underlying market data to condition the diffusion process.

**Acknowledgements**

Special thanks to  Professor Stefano Ermon at Stanford University for his advice and to Elyas Obbad for mentorship throughout this project.






