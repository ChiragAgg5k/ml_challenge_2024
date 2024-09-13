# model_trainer.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# Import constants
from constants import entity_unit_map

# Assuming you have a GPU available. If not, change to 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_image(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def extract_features(image):
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.squeeze().cpu().numpy()

def load_vit_model():
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    return feature_extractor, model

def extract_vit_features(image, feature_extractor, model):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze().cpu().numpy()

def extract_all_features(image_link):
    try:
        image = download_image(image_link)
        resnet_features = extract_features(image)
        vit_feature_extractor, vit_model = load_vit_model()
        vit_features = extract_vit_features(image, vit_feature_extractor, vit_model)
        return np.concatenate([resnet_features, vit_features])
    except Exception as e:
        print(f"Error processing {image_link}: {str(e)}")
        return None

def validate_entity_value(row):
    entity_name = row['entity_name']
    entity_value = row['entity_value']
    
    # Split the value and unit
    parts = entity_value.split()
    if len(parts) != 2:
        return False
    
    value, unit = parts
    
    # Check if the unit is allowed for this entity
    if entity_name in entity_unit_map and unit in entity_unit_map[entity_name]:
        try:
            float(value)  # Check if the value can be converted to float
            return True
        except ValueError:
            return False
    return False

def train_model(train_data):
    X = []
    y = []
    
    print("Validating and extracting features from images...")
    for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Processing Images"):
        if validate_entity_value(row):
            features = extract_all_features(row['image_link'])
            if features is not None:
                X.append(features)
                y.append(row['entity_value'])
        else:
            print(f"Invalid entity value: {row['entity_name']} - {row['entity_value']}")
    
    X = np.array(X)
    y = np.array(y)
    
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)  # Reduced n_estimators for faster training
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_val)
    
    unique_classes = np.unique(y_val)    
    target_names = le.inverse_transform(np.unique(y_encoded))  # All possible classes

    print(classification_report(y_val, y_pred, target_names=target_names, labels=unique_classes))
    
    return clf, le

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    MODEL_FOLDER = 'models/'
    
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    
    print("Loading training data...")
    train_data = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    train_data = train_data.head(10)  # Only use the first 50 samples for testing
    
    print(f"Training data loaded. Total samples: {len(train_data)}")
    print(f"Unique entities in this sample: {train_data['entity_name'].nunique()}")
    print(f"Unique entity values in this sample: {train_data['entity_value'].nunique()}")
    
    trained_model, label_encoder = train_model(train_data)
    
    print("Saving model and label encoder...")
    joblib.dump(trained_model, os.path.join(MODEL_FOLDER, 'trained_model.joblib'))
    joblib.dump(label_encoder, os.path.join(MODEL_FOLDER, 'label_encoder.joblib'))
    
    print("Model and label encoder saved successfully.")
    print("Training complete!")