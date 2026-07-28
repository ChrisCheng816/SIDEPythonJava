from sentence_transformers import SentenceTransformer, models, InputExample, losses, util, LoggingHandler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import torch.nn.functional as F
import json
from sentence_transformers import evaluation
from tqdm import tqdm
import logging


logging.basicConfig(format='%(asctime)s - %(message)s',
					datefmt='%Y-%m-%d %H:%M:%S',
					level=logging.INFO,
					handlers=[LoggingHandler()])



def train(train_data, eval_data=None):

	n_train_examples = len(train_data)

	train_examples = []

	for i in range(n_train_examples):
		example = train_data[i]

		train_examples.append(InputExample(
			texts=[example['query'], example['pos'], example['neg']]))

		if example['hardNegative'] != []:
			for negative in example['hardNegative']:
				train_examples.append(InputExample(texts=[example['query'], example['pos'], negative]))

	train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

	# Step 1: use an existing language model
	word_embedding_model = models.Transformer('microsoft/mpnet-base')

	# Step 2: use a pool function over the token embeddings
	pooling_model = models.Pooling(
		word_embedding_model.get_word_embedding_dimension())

	# Join steps 1 and 2 using the modules argument
	model=SentenceTransformer(modules = [word_embedding_model, pooling_model])

	train_loss=losses.TripletLoss(model = model)

	num_epochs=10
	warmup_steps=int(len(train_dataloader) *
					   num_epochs * 0.1)  # 10% of train data

	model.fit(train_objectives = [(train_dataloader, train_loss)],
			  epochs=num_epochs,
			  checkpoint_save_steps=5000,
			  checkpoint_path="models/triplet-loss/hard_negatives",
			  output_path="models/triplet-loss/hard_negatives",
			  warmup_steps=warmup_steps,
			  show_progress_bar=True)


def main():
	with open('dataset/train.json') as f:
		train_data = json.load(f)

	train(train_data)

if __name__ == "__main__":
	main()
