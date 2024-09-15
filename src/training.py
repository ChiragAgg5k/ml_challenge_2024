import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from constants import entity_unit_map


# Import constants

# Assuming you have a GPU available. If not, change to 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Check if the image is corrupted by trying to load it
            img.verify()
        # If verification passes, open and return the image
        return Image.open(image_path).convert("RGB")
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None


def extract_features(image):
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor)

    return features.squeeze().cpu().numpy()


def load_vit_model():
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(
        device
    )
    return image_processor, model


def extract_vit_features(image, image_processor, model):
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze().cpu().numpy()


def extract_all_features(image_path):
    try:
        image = load_image(image_path)
        if image is None:
            return None
        resnet_features = extract_features(image)
        vit_feature_extractor, vit_model = load_vit_model()
        vit_features = extract_vit_features(image, vit_feature_extractor, vit_model)
        return np.concatenate([resnet_features, vit_features])
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def validate_entity_value(row):
    entity_name = row["entity_name"]
    entity_value = row["entity_value"]

    parts = entity_value.split()
    if len(parts) != 2:
        return False

    value, unit = parts

    if entity_name in entity_unit_map and unit in entity_unit_map[entity_name]:
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def process_image(row, images_folder):
    if validate_entity_value(row):
        image_filename = os.path.basename(row["image_link"])
        image_path = os.path.join(images_folder, image_filename)
        features = extract_all_features(image_path)
        if features is not None:
            return features, row["entity_value"]
    return None


def train_model(train_data, images_folder):
    X = []
    y = []

    print("Validating and extracting features from images...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_row = {
            executor.submit(process_image, row, images_folder): row
            for _, row in train_data.iterrows()
        }
        for future in tqdm(
            as_completed(future_to_row), total=len(train_data), desc="Processing Images"
        ):
            result = future.result()
            if result is not None:
                features, entity_value = result
                X.append(features)
                y.append(entity_value)

    if not X:
        print("No valid images were processed. Cannot train the model.")
        return None, None

    X = np.array(X)
    y = np.array(y)

    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_val)

    unique_classes = np.unique(y_val)
    target_names = le.inverse_transform(unique_classes)

    print(
        classification_report(
            y_val, y_pred, target_names=target_names, labels=unique_classes
        )
    )

    return clf, le


if __name__ == "__main__":
    DATASET_FOLDER = "dataset/"
    MODEL_FOLDER = "models/"
    IMAGES_FOLDER = "images/"

    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    print("Loading training data...")
    train_data = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))

    print(f"Training data loaded. Total samples: {len(train_data)}")
    print(f"Unique entities: {train_data['entity_name'].nunique()}")
    print(f"Unique entity values: {train_data['entity_value'].nunique()}")

    trained_model, label_encoder = train_model(train_data, IMAGES_FOLDER)

    if trained_model is not None and label_encoder is not None:
        print("Saving model and label encoder...")
        joblib.dump(trained_model, os.path.join(MODEL_FOLDER, "trained_model.joblib"))
        joblib.dump(label_encoder, os.path.join(MODEL_FOLDER, "label_encoder.joblib"))
        print("Model and label encoder saved successfully.")
        print("Training complete!")
    else:
        print("Training failed due to lack of valid data.")
