# from dziner.surrogates.MOFormer.model.transformer import Transformer, TransformerRegressor # type: ignore
import yaml
import torch
import numpy as np
import sys
import os


folder_path = '../dziner/surrogates/MOFormer/' 
sys.path.append(folder_path)

if folder_path not in sys.path:
    sys.path.append(folder_path)


config = yaml.load(open(os.path.join(folder_path,("config_ft_transformer.yaml")), "r"), Loader=yaml.FullLoader)

if torch.cuda.is_available() and config['gpu'] != 'cpu':
    device = config['gpu']
    torch.cuda.set_device(device)
    config['cuda'] = True


from tokenizer.mof_tokenizer import MOFTokenizer
tokenizer = MOFTokenizer(os.path.join(folder_path,"tokenizer/vocab_full.txt"))

# inference_model = torch.load(os.path.join(folder_path,'hmof_finetuned_models/hmof_finetuned_0.pth'))
# inference_model.to(device)

def SMILES_to_CO2_adsorption(smiles, model):
    token = np.array([tokenizer.encode(smiles, max_length=512, truncation=True,padding='max_length')])
    token = torch.from_numpy(np.asarray(token))
    token = token.to(device)
    return model(token).item()


def predict_CO2_adsorption_with_uncertainty(smiles):
    predictions = []
    for fold in range(5):
        inference_model = torch.load(os.path.join(folder_path,f'hmof_finetuned_models/hmof_finetuned_{fold}.pth'))
        inference_model.to(device)
        predictions.append(SMILES_to_CO2_adsorption(smiles, model=inference_model))
    return np.mean(predictions), np.std(predictions)