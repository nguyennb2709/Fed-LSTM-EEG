import pandas as pd
import numpy as np
import os
import time  # Import the time module
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset  # Import DataLoader

# Define the fixed input path to your dataset folder
input_path = "Dataset_Original"

# Initialize lists to hold the samples and their corresponding labels
samples = []
labels = []
log_dir = '/content/drive/My Drive/eeg_logs'
writer = SummaryWriter(log_dir)
# Ensure the directory exists
os.makedirs(log_dir, exist_ok=True)
# Load EEG data from CSV files
for i in range(1, 89):  # Assuming files are named 001.csv to 088.csv
    file_path = os.path.join(input_path, f"{i:03}.csv")
    print(f"Loading file: {file_path}")

    # Load the CSV without headers and drop the first column
    df = pd.read_csv(file_path, header=None, low_memory=False)
    df = df.iloc[:, 1:]  # Keep all rows, drop the first column (time or index column)

    # Convert all values in the DataFrame to numeric, forcing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Optionally drop rows with NaN values

    # Check if the file has any data left
    if df.empty:
        print(f"Warning: File {i:03}.csv is empty after preprocessing.")
        continue  # Skip empty files

    # Convert the DataFrame to a numpy array and append to samples
    samples.append(df.values)

    # Assign labels based on the file number
    if i <= 36:
        label = 0  # Class 1
    elif i <= 65:
        label = 1  # Class 2
    else:
        label = 2  # Class 3
    labels.extend([label] * len(df))  # Replicate the label for all rows in the current file

    print(f"File {i:03}.csv loaded: {df.shape[0]} samples, {df.shape[1]} features.")

# Convert samples and labels to PyTorch tensors
samples = torch.tensor(np.vstack(samples), dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)  # Labels now in the range 0, 1, 2
print(f"Total samples shape: {samples.shape}, Total labels shape: {labels.shape}")

# Normalize the input data
scaler = StandardScaler()
samples_np = samples.numpy()
samples_np = scaler.fit_transform(samples_np)  # Fit and transform the data
samples = torch.tensor(samples_np, dtype=torch.float32)  # Convert back to tensor

# Reshape the data for LSTM input: (samples, sequence_length, features)
sequence_length = 10  # Adjust this value based on your data
num_features = samples.shape[1]

# Check if samples can be reshaped
num_samples = samples.shape[0]

if num_samples % sequence_length != 0:
    # Trim the samples and labels to ensure divisibility
    trim_size = num_samples - (num_samples % sequence_length)
    samples = samples[:trim_size]
    labels = labels[:trim_size]

# Reshape the samples for LSTM
samples_lstm = samples.view(-1, sequence_length, num_features)

# Take labels and reshape them accordingly (ensuring one label per sample)
labels_lstm = labels.view(-1, sequence_length)

# Ensure labels correspond to the last time step
labels_lstm = labels_lstm[:, -1]  # Keep only the last label for each sequence

# Now, check shapes again
print(f"LSTM input shape: {samples_lstm.shape}, Labels shape: {labels_lstm.shape}")

# Split the dataset into training (70%) and temporary set (30%)
X_temp, X_test, y_temp, y_test = train_test_split(
    samples_lstm,
    labels_lstm,
    test_size=0.3,
    stratify=labels_lstm,  # Make sure stratification uses correct labels
    random_state=42
)
# Now split the temporary set into training (70% of total) and evaluation (10% of total)
X_train, X_eval, y_train, y_eval = train_test_split(
    X_temp,
    y_temp,
    test_size=0.3333,  # 10% of original data (30% * 1/3)
    stratify=y_temp,  # Stratify the evaluation split
    random_state=42
)

num_samples, sequence_length, num_features = X_train.shape
X_train_reshaped = X_train.reshape(num_samples * sequence_length, num_features)  # Reshape to (total_samples, num_features)
y_train_reshaped = y_train.repeat(sequence_length)   # Should give (number_of_samples,)

print(f"After splitting and reshaping: X_train_reshaped shape: {X_train_reshaped.shape}, y_train_reshaped shape: {y_train_reshaped.shape}")
# Ensure y_train_reshaped matches the length of X_train_reshaped
if y_train_reshaped.shape[0] != X_train_reshaped.shape[0]:
    raise ValueError(f"y_train_reshaped length {y_train_reshaped.shape[0]} does not match X_train_reshaped length {X_train_reshaped.shape[0]}.")

## Now apply SMOTE
print("Applying SMOTE...")
start_time = time.time()
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped.numpy(), y_train_reshaped.numpy())  # Convert to numpy for SMOTE
end_time = time.time()
print(f"Time taken for SMOTE: {end_time - start_time:.4f} seconds")

# Print the shapes of the original and SMOTE-applied datasets
print(f"Original training data shape: {X_train_reshaped.shape}, Original labels shape: {y_train_reshaped.shape}")
print(f"SMOTE training data shape: {X_train_smote.shape}, SMOTE labels shape: {y_train_smote.shape}")

# Convert augmented data back to tensors
X_train_smote = torch.tensor(X_train_smote, dtype=torch.float32)
y_train_smote = torch.tensor(y_train_smote, dtype=torch.long)

# Reshape the oversampled data back into the LSTM input shape
X_train_smote_lstm = X_train_smote.reshape(-1, sequence_length, num_features)

# Verify class distribution after SMOTE
unique_smote, counts_smote = np.unique(y_train_smote.numpy(), return_counts=True)
print(f"Class distribution after SMOTE: {dict(zip(unique_smote, counts_smote))}")

# Data augmentation function
def augment_data(X, y):
    # Add noise
    noise = np.random.normal(0, 0.001, X.shape)  # Adjust noise level as needed
    X_augmented = X + noise

    # Time shifting (for example, shift left)
    shift = np.random.randint(1, 3)  # Randomly shift between 1 to 5 samples
    X_shifted = np.roll(X, shift, axis=1)  # Shift along time dimension (last axis)

    # Scaling
    scale_factor = np.random.uniform(0.95, 1.05)  # Scale between 90% to 110%
    X_scaled = X * scale_factor

    # Stack augmented data
    X_combined = np.vstack([X, X_augmented, X_shifted, X_scaled])
    y_combined = np.hstack([y, y, y, y])  # Repeat labels for new samples

    print(f"Augmented data shape: {X_combined.shape}, Augmented labels shape: {y_combined.shape}")
    return X_combined, y_combined
print('Begin augmentation')
# Augment training data
X_train_augmented, y_train_augmented = augment_data(X_train.numpy(), y_train.numpy())
print('End augmentation')
# Convert augmented data back to tensors
X_train_augmented = torch.tensor(X_train_augmented, dtype=torch.float32)
y_train_augmented = torch.tensor(y_train_augmented, dtype=torch.long)

# Define the LSTM model with dropout and batch normalization
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.batch_norm = nn.BatchNorm1d(128)  # Batch normalization layer
        self.fc = nn.Linear(128, 3)  # Output layer for 3 classes (bidirectional doubles the hidden size)
        self.dropout = nn.Dropout(0.3)  # Additional dropout layer after the LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Get LSTM output
        out = self.batch_norm(lstm_out[:, -1, :])  # Apply batch normalization to the last LSTM output
        out = self.fc(self.dropout(out))  # Use the output with dropout for classification
        return out

# Calculate the total number of samples after SMOTE
total_samples_smote = sum(counts_smote)

# Calculate class weights based on the new class distribution
class_weights_smote = torch.tensor(
    [total_samples_smote / (len(unique_smote) * count) for count in counts_smote],
    dtype=torch.float32  # Ensure weights are Float
)

# Initialize model, loss function, and optimizer
model = LSTMModel()
criterion = nn.CrossEntropyLoss(weight=class_weights_smote)  # Use class weights in the loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print('Initialize model')
# Create DataLoader for training and evaluation datasets
train_dataset = TensorDataset(X_train_augmented, y_train_augmented)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

eval_dataset = TensorDataset(X_eval, y_eval)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

# Training loop with TensorBoard logging
num_epochs = 3 # Number of epochs for training
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        running_loss += loss.item()  # Accumulate loss

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)  # Log training loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Testing loop remains unchanged
model.eval()  # Set the model to evaluation mode
y_pred = []
y_true = []
y_prob = []  # To store probabilities

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        y_pred.extend(predicted.numpy())
        y_true.extend(batch_y.numpy())
        
        # Store the probabilities for each class
        y_prob.extend(torch.softmax(outputs, dim=1).numpy())  # Convert logits to probabilities

# Compute metrics for the test set
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# Use the predicted probabilities for average precision score
average_precision = average_precision_score(y_true, y_prob, average='macro')

print(f"Test Set - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Average Precision: {average_precision:.4f}")

# Evaluation loop
model.eval()  # Set the model to evaluation mode for evaluation set
y_eval_pred = []
y_eval_true = []
y_eval_prob = []  # To store probabilities for evaluation set

with torch.no_grad():
    for batch_X, batch_y in eval_loader:
        outputs = model(batch_X)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        y_eval_pred.extend(predicted.numpy())
        y_eval_true.extend(batch_y.numpy())
        
        # Store the probabilities for each class
        y_eval_prob.extend(torch.softmax(outputs, dim=1).numpy())  # Convert logits to probabilities

# Compute metrics for the evaluation set
eval_accuracy = accuracy_score(y_eval_true, y_eval_pred)
eval_precision = precision_score(y_eval_true, y_eval_pred, average='weighted', zero_division=0)
eval_recall = recall_score(y_eval_true, y_eval_pred, average='weighted', zero_division=0)
eval_f1 = f1_score(y_eval_true, y_eval_pred, average='weighted', zero_division=0)

# Use the predicted probabilities for average precision score
eval_average_precision = average_precision_score(y_eval_true, y_eval_prob, average='macro')

print(f"Evaluation Set - Accuracy: {eval_accuracy:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}, F1 Score: {eval_f1:.4f}, Average Precision: {eval_average_precision:.4f}")

# Save the model remains unchanged
torch.save(model.state_dict(), 'lstm_model.pth')

# Close TensorBoard writer remains unchanged
writer.close()