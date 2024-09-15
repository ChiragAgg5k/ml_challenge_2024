# main.py

import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


# Assuming you have a GPU available. If not, change to 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_image(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content)).convert("RGB")


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


def extract_all_features(image_link):
    try:
        image = download_image(image_link)
        resnet_features = extract_features(image)
        image_processor, vit_model = load_vit_model()
        vit_features = extract_vit_features(image, image_processor, vit_model)
        return np.concatenate([resnet_features, vit_features])
    except Exception as e:
        print(f"Error processing {image_link}: {str(e)}")
        return None


def predictor(image_link, category_id, entity_name, model, label_encoder):
    features = extract_all_features(image_link)
    if features is not None:
        prediction = model.predict([features])[0]
        return label_encoder.inverse_transform([prediction])[0]
    else:
        return ""


if __name__ == "__main__":
    DATASET_FOLDER = "dataset/"
    MODEL_FOLDER = "models/"

    # Load the trained model and label encoder
    trained_model = joblib.load(os.path.join(MODEL_FOLDER, "trained_model.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_FOLDER, "label_encoder.joblib"))

    # Load test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_test.csv"))
    test = test.head(10)  # Only use the first 50 samples for testing

    # Make predictions with progress bar
    tqdm.pandas(desc="Making Predictions")  # Set up tqdm for pandas
    test["prediction"] = test.progress_apply(
        lambda row: predictor(
            row["image_link"],
            row["group_id"],
            row["entity_name"],
            trained_model,
            label_encoder,
        ),
        axis=1,
    )

    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, "test_out.csv")
    test[["index", "prediction"]].to_csv(output_filename, index=False)

    print("Predictions saved successfully.")
