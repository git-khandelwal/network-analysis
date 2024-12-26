from model import SmallNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pipe import preprocessing_pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
from dataset import MyDataset
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv(r'log2.csv')
y = df['Action']
x = df.drop(columns='Action')

x_transformed = preprocessing_pipeline.fit_transform(x)
joblib.dump(preprocessing_pipeline, "my_pipeline.pkl")

le = LabelEncoder()
y_labelencoded = le.fit_transform(np.array(y))

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y_labelencoded, test_size=0.25, random_state=42)

# Create datasets and dataloaders for training and testing
train_dataset = MyDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = MyDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = SmallNet(input_size=19, hidden_size=64, output_size=4)

num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(num_epochs):
    for x, y in train_dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_dataloader:
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
        all_predictions.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Print precision, recall, and F1 score
print(classification_report(all_labels, all_predictions))

# Save the model weights
torch.save(model.state_dict(), "trained_weights.pth")
    