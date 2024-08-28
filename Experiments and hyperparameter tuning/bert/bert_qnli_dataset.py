import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


dataset = load_dataset("glue", "qnli")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['question'], examples['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename the label column to 'labels' which is expected by the model
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set the format of the datasets
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets['train'].select(range(1000))
eval_dataset = tokenized_datasets['validation']

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
eval_dataloader = DataLoader(eval_dataset, batch_size=64)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

# Evaluation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy}")
