from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

def calculate_bleu_score(generated_text, reference_text):
    generated_tokens = word_tokenize(generated_text)
    reference_tokens = word_tokenize(reference_text)
    return sentence_bleu([reference_tokens], generated_tokens)

# Example usage
generated_text = "This is an example sentence."
reference_text = "This is a reference sentence."

bleu_score = calculate_bleu_score(generated_text, reference_text)
print(f"BLEU Score: {bleu_score:.4f}") 
