import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize import word_tokenize

def calculate_perplexity(logits, targets):
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()


def preprocess_function(text):
    tokens = word_tokenize(text)  # Tokenization using NLTK
    # Other preprocessing steps (e.g., lowercasing, padding, numerical encoding) can be added here
    return tokens 
def calculate_accuracy(logits, targets):
    _, predicted = torch.max(logits, dim=2)
    correct = (predicted == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total
    return accuracy

# Example language model class
class MyLanguageModel(nn.Module):
    def __init__(self):
        super(MyLanguageModel, self).__init__()
        # Your language model implementation goes here

# Example usage
model_path = '/home/srib/aryan4/models/koala-7b.bin'  # Replace with your actual model path

generated_text = "This is an example sentence."
reference_text = "This is a reference sentence."

# Example preprocessing steps (tokenization, encoding) to convert texts to numerical representation
# Replace with your own implementation
generated_input = preprocess_function(generated_text)
reference_target = preprocess_function(reference_text)

# Convert input and target to tensors
inputs = torch.tensor(generated_input)
targets = torch.tensor(reference_target)

inputs = inputs.unsqueeze(0)  # Add batch dimension
targets = targets.unsqueeze(0)  # Add batch dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyLanguageModel()  # Replace with your language model implementation
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

inputs = inputs.to(device)
targets = targets.to(device)

model.eval()

with torch.no_grad():
    logits = model(inputs)
    perplexity = calculate_perplexity(logits, targets)
    accuracy = calculate_accuracy(logits, targets)

print(f"Perplexity: {perplexity:.4f}")
print(f"Accuracy: {accuracy:.4f}") 
