from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

llm_response = 'Machine learning is a subset of artificial intelligence (AI) that focuses on the evelopment of algorithms and statistical models that enable computers to perform tasks without being explicitly programmed. In traditional programming, a programmer writes explicit instructions for the computer to follow, whereas in machine learning, the computer learns from data provided to it.The core idea behind machine learning is to enable computers to learn from data patterns and make decisions or predictions based on that learning. This process involves training a model on a dataset, which typically includes input data (features) and corresponding output labels. The model learns the underlying patterns in the data and can then make predictions or decisions when given new, unseen data.'
user_response = 'i dont know about machine learning'

# Sentences we want sentence embeddings for
sentences = [llm_response,user_response]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

similarity = cosine_similarity(
    sentence_embeddings[0].reshape(1, -1),
    sentence_embeddings[1].reshape(1, -1)
)[0][0]

print("similarity:", similarity) 

