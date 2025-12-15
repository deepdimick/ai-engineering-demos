## models/image_processor.py

# Import necessary libraries
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import requests
import base64
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImageProcessor:
    """
    Handles image processing, encoding, and similarity comparisons.
    """
    # Initialize the model and preprocessing pipeline
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load V2 weights and use their recommended transforms
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights).to(self.device)
        self.model.eval()
        
        # Use the transforms that match these weights
        self.preprocess = self.weights.transforms()

    # Encode an image and return its Base64 representation and feature vector
    def encode_image(self, image_input, is_url=True):
        # Handle image input based on whether it's a URL or local file
        try:
            if is_url:
                # Fetch the image from URL
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Load the image from a local file
                image = Image.open(image_input).convert("RGB")

            # Convert image to Base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Preprocess the image for ResNet50
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            # Extract features using ResNet50
            with torch.no_grad():
                features = self.model(input_tensor)
            # Convert features to NumPy array
            feature_vector = features.cpu().numpy().flatten()
            
            return {"base64": base64_string, "vector": feature_vector}
        # Handle exceptions during image processing
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}

    # Find the closest match in a dataset based on cosine similarity
    def find_closest_match(self, user_vector, dataset):
        try:
            dataset_vectors = np.vstack(dataset['Embedding'].dropna().values)
            similarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)
            
            # Find the index of the most similar vector
            closest_index = np.argmax(similarities)
            similarity_score = similarities[0][closest_index]
            
            # Retrieve the closest matching row
            closest_row = dataset.iloc[closest_index]
            return closest_row, similarity_score
        # Handle exceptions during similarity calculation
        except Exception as e:
            print(f"Error finding closest match: {e}")
            return None, None