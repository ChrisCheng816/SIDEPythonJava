# Evaluating Code Summarization Techniques: A New Metric and an Empirical Characterization

In this study, we explore the suitability of current state-of-the-art code summarization metrics for evaluating the similarity between two pieces of text: the reference and the generated summary. Traditionally, this evaluation is based on comparing the word tokens in the reference with those predicted by a code summarizer. However, we identify two scenarios where this assumption may fall short:

(i) The reference summary, often extracted from code comments in software repositories, may be of low quality due to outdated or less relevant information.

(ii) The generated summary might use different wording compared to the reference but still convey the same meaning, making it suitable to document the code snippet effectively.

To address these limitations, we introduce a new dimension that assesses the extent to which the generated summary aligns with the semantics of the documented code snippet independently of the reference summary. By empirically evaluating this aspect, we propose a new metric called SIDE (Summary alIgnment to coDe sEmantics) that leverages contrastive learning to capture better the developers' assessment of the automatically generated summaries' quality. 

#### Pipeline Description

To build SIDE (Summary alIgnment to coDe sEmantics), we relied on Contrastive learning to learn an embedding space where similar sample pairs (i.e., pairs sharing specific features) are clustered together while dissimilar pairs are set apart. In our context, we want to use contrastive learning to teach the model to discriminate textual summaries suitable for a given code snippet vs. unsuitable summaries.


#### Contrastive Loss
In our study, we employ the Triplet loss, which has been shown to better encode the positive/negative samples as compared to other contrastive losses. The triplet loss function has been proposed by Schroff et. al  and introduces the concept of "anchor". Given an anchor $x$, a positive ($x^{+}$) and a negative ($x^{-}$) sample is selected, with the triplet loss which during training minimizes the distance between the $x$ and $x^{+}$, while maximizing the distance between $x$ and $x^{-}$:

$$
	E=\max \left(\left\|f_a-f_p\right\|^2-\left\|f_a-f_n\right\|^2+m, 0\right)
$$

In our case, the anchor is the code to document, with a suitable summary representing $x^{+}$ and an unsuitable summary representing $x^{-}$. In the following sections, we introduce the dataset used to fine-tune MPNet for the task of interest, explaining how we generate positive and negative samples.


![image info](triplet-example.png)



* ##### Datasets :paperclip:

    The datasets used for training and evaluation are available <a href="https://drive.google.com/drive/folders/1jbXHcPoy-S4BeMhDL57MDHcNzBuXHOB8?usp=sharing">here</a>
    
    
* ##### Fine-tuned Models :computer:
    The model we trained to develop SIDE is publicly available at the following link: <a href ='https://drive.google.com/drive/folders/150xbvYtyuUNsd8hefjiZXqa_eFY8f67K?usp=sharing'>Models</a> 


* ##### Statistical Analysis
    The scripts needed to conduct the analysis are available <a href='https://github.com/antonio-mastropaolo/code-summarization-metric/tree/main/Analysis'>here</a> 


* ##### How to use SIDE
   1. Create a new virtual env using: ```python3 -m venv venv``` 
   2. Activate the new env: ```source venv/bin/activate```
   3. Install all the dependencies using pip: ```pip install -r requirements.txt```
   4. Run the following script:
  ```
    from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
    from transformers import AutoTokenizer, AutoModel
    import torch
    import sys,os
    import torch.nn.functional as F
    from sentence_transformers import evaluation
    from tqdm import tqdm

    DEVICE = "cpu"

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    checkPointFolder = "models/triplet-loss/..." #specify the path to the best-performing checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
    model = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)

    method = """
		public Object pop() throws EmptyStackException {
		    try {
		      Object aObject = this.stackElements.get(stackElements.size() - 1);
		      stackElements.remove(stackElements.size() - 1);
		      return aObject;
		    }
		    catch (Exception e) {
		      throw new EmptyStackException(e);
		    }
  		}
    """
    codeSummary = "pop the top of the stack"

    pair = [method,codeSummary]

    encoded_input = tokenizer(
            pair, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    sim = util.pytorch_cos_sim(
        sentence_embeddings[0], sentence_embeddings[1]).item()

    print(sim)
    #0.836042046546936

  ```
    



    
