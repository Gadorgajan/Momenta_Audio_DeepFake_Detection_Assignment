import os
import torch
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from torch.utils.data import DataLoader, Dataset as TorchDataset
import soundfile as sf
import librosa
from sklearn.metrics import roc_curve
from gtts import gTTS

import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define dataset path
DATASET_PATH = "dataset/protocol_V2"
SUBFOLDERS = ["ASVspoof2017_V2_dev", "ASVspoof2017_V2_eval"]  # Add 'protocol_V2' if it contains files

# List to store file details
audio_data = []

# Recursively load all .wav files
for subfolder in SUBFOLDERS:
    folder_path = os.path.join(DATASET_PATH, subfolder)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):  # Only process audio files
                file_path = os.path.join(root, file)
                
                # Load audio file
                audio, sr = librosa.load(file_path, sr=16000)
                
                # Extract features (Mel Spectrogram)
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
                mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

                # Store data
                audio_data.append({
                    "filename": file,
                    "filepath": file_path,
                    "duration": librosa.get_duration(y=audio, sr=sr),
                    "sampling_rate": sr,
                    "mel_spectrogram": mel_spec_db
                })

# Convert to DataFrame
df = pd.DataFrame(audio_data)

# Display dataset summary
print(df.head())

# 🔹 Example: Visualizing a Mel spectrogram of the first file
plt.figure(figsize=(10, 4))
librosa.display.specshow(df["mel_spectrogram"][0], sr=16000, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title(f"Mel Spectrogram: {df['filename'][0]}")
plt.show()

# Load processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Custom Dataset
class AudioDataset(TorchDataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]["filepath"]
        label = self.df.iloc[idx]["label"]
        audio, sr = sf.read(audio_path)  # Use soundfile for faster loading
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if np.random.rand() > 0.5:  # Random noise augmentation
            audio += np.random.normal(0, 0.005, len(audio))
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values.squeeze()
        return {"input_values": inputs, "labels": torch.tensor(label)}

# Split into train and eval (80-20 split)
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

# Create datasets and loaders
train_dataset = AudioDataset(train_df, processor)
eval_dataset = AudioDataset(eval_df, processor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add label column to the DataFrames
df['label'] = df['filename'].apply(lambda x: 1 if 'spoof' in x.lower() else 0)
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# Split into train and eval (80-20 split)
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

# Create datasets
train_dataset = AudioDataset(train_df, processor)
eval_dataset = AudioDataset(eval_df, processor)

# Custom collation function to handle variable length inputs
def collate_batch(batch):
    # Get input values and labels from batch
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Use processor to pad the inputs to the same length
    input_values = processor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt",
    )
    
    # Stack labels
    labels = torch.stack(labels)
    
    return {
        "input_values": input_values.input_values,
        "labels": labels,
    }

# Create data loaders with custom collation function
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

# Load model
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Slightly higher lr for faster convergence
num_epochs = 5  # More epochs for better training

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        inputs = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

# Save model and processor
model.save_pretrained("./saved_wav2vec2_fraud_detector")
processor.save_pretrained("./saved_wav2vec2_fraud_detector")
print("Model and processor saved to './saved_wav2vec2_fraud_detector/'")

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[eer_idx]

model.eval()
all_labels = []
all_scores = []
with torch.no_grad():
    for batch in eval_loader:
        inputs = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()  # Spoof probability
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores)

eer = compute_eer(all_labels, all_scores)
print(f"Equal Error Rate (EER): {eer*100:.2f}%")

def detect_fraud(audio_path, model, processor):
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        prob_spoof = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
    return "Fraud (Spoof)" if prob_spoof > 0.5 else "Genuine (Bonafide)", prob_spoof

# Test inference
sample_audio = df["filepath"].iloc[0]
label, prob = detect_fraud(sample_audio, model, processor)
print(f"Prediction for {sample_audio}: {label} (Probability of Fraud: {prob:.4f})")
