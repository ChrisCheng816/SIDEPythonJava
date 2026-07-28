from sentence_transformers import SentenceTransformer, models, InputExample, losses, util, LoggingHandler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import sys,os
import torch.nn.functional as F
import json
from sentence_transformers import evaluation
from tqdm import tqdm
import logging

DEVICE='cuda'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def find_digit_folders(path):
    digit_folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)) and folder.isdigit():
            digit_folders.append(int(folder))
    sorted_list = sorted(digit_folders)
    return sorted_list


def test(test_data, model_path):
    
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Example usage
    digit_folders = find_digit_folders(model_path)
    

    checkpointsList = []
    cosineSimilaritiesList = []

    for folder in digit_folders:

        checkPointFolder = os.path.join(model_path, str(folder))
        

        tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
        model = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)
        similarities = []
        overallScore = 0

        print("*****************************************************************")
        print(f"[+] Evaluating Model Checkpoint: {folder}")

        for item in tqdm(test_data):

            ################## POSITIVE ##################

            sentences = []
            sentences.append(item['query'])
            sentences.append(item['pos'])

            # Tokenize sentences
            encoded_input = tokenizer(
                sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            simPositive = util.pytorch_cos_sim(
                sentence_embeddings[0], sentence_embeddings[1]).item()
            
            #######################################################################

            ################## NEGATIVE ##################
            sentences = []
            sentences.append(item['query'])
            sentences.append(item['neg'])

            
            # Tokenize sentences
            encoded_input = tokenizer(
                sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            simNegatives = util.pytorch_cos_sim(
                sentence_embeddings[0], sentence_embeddings[1]).item()

            overallScore += simPositive - simNegatives
        
        print(f"Overall Score at checkpoint {folder} = {overallScore/len(test_data)} ")
        print("*****************************************************************\n")
        checkpointsList.append(folder)
        cosineSimilaritiesList.append(overallScore/len(test_data))
    
    df_eval_res = pd.DataFrame(zip(checkpointsList,cosineSimilaritiesList),columns=['checkpoint','cosine_sim'])
    df_eval_res.to_csv(os.path.join(sys.argv[1],"evalResults.csv"))

def main():

    with open('dataset/eval.json') as f:
        eval_data = json.load(f)
    
    test(eval_data, sys.argv[1])


if __name__ == "__main__":
    main()
