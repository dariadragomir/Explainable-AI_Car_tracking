import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import precision_score, recall_score
from datasets import load_dataset

dataset = load_dataset("glue", "rte")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset_1 = tokenized_datasets['train'].select(range(800, 810))  # Range 800-1000
train_dataset_2 = tokenized_datasets['train'].select(range(1000, 1010)) # Range 1000-1200
'''
# Function to plot text length distribution
def plot_text_length_distribution(dataset, sentence_column):
    lengths = [len(tokenizer.encode(s)) for s in dataset[sentence_column]]
    plt.figure(figsize=(10,6))
    sns.histplot(lengths, kde=True, bins=30)
    plt.title(f'Text Length Distribution ({sentence_column})')
    plt.xlabel('Length (Tokens)')
    plt.ylabel('Frequency')
    plt.show()

# Plot text length distribution for sentence1 and sentence2 in range (800, 1000)
print("Text Length Distribution for Range (800-1000):")
plot_text_length_distribution(train_dataset_1, 'sentence1')
plot_text_length_distribution(train_dataset_1, 'sentence2')

# Plot text length distribution for sentence1 and sentence2 in range (1000, 1200)
print("Text Length Distribution for Range (1000-1200):")
plot_text_length_distribution(train_dataset_2, 'sentence1')
plot_text_length_distribution(train_dataset_2, 'sentence2')
'''
for elem in train_dataset_2['sentence1']:
    print(elem)
    print()

for elem in train_dataset_2['sentence2']:
    print(elem)    
    print()
print(train_dataset_1)
print(train_dataset_2)
