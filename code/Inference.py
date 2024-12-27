import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def Net():
    """Define the model architecture for dog breed classification."""
    model = models.resnet50(pretrained=True)

    # Freeze all layers except the fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 133)  # 133 classes for dog breeds
    )
    return model

def model_fn(model_dir):
    """Load the model from the specified directory.

    Args:
        model_dir (str): Path to the directory containing the model.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model ready for inference.
    """
    logger.info("Loading the model from the directory.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    # Load the model state dictionary
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info("Model loaded successfully.")
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    """Process input data for prediction.

    Args:
        request_body (bytes): The request body sent to the endpoint.
        content_type (str): The MIME type of the input data.

    Returns:
        PIL.Image.Image: Processed input image.

    Raises:
        ValueError: If the content type is unsupported.
    """
    logger.info("Deserializing the input data.")

    if content_type == JPEG_CONTENT_TYPE:
        logger.debug("JPEG content type detected.")
        return Image.open(io.BytesIO(request_body))

    if content_type == JSON_CONTENT_TYPE:
        logger.debug("JSON content type detected.")
        request = json.loads(request_body)
        url = request["url"]
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))

    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_object, model):
    """Generate predictions from the input object.

    Args:
        input_object (PIL.Image.Image): Input image to classify.
        model (torch.nn.Module): Loaded PyTorch model.

    Returns:
        torch.Tensor: Raw predictions from the model.
    """
    logger.info("Generating prediction.")

    # Define the transformation pipeline
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Transform the input image
    input_object = test_transform(input_object).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = model(input_object)

    return prediction

def output_fn(prediction, content_type=JSON_CONTENT_TYPE):
    """Format the prediction output for the client.

    Args:
        prediction (torch.Tensor): Raw predictions from the model.
        content_type (str): Desired content type for the response.

    Returns:
        str: JSON-formatted prediction results.

    Raises:
        ValueError: If the content type is unsupported.
    """
    logger.info("Formatting prediction output.")

    if content_type == JSON_CONTENT_TYPE:
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0).tolist()
        top_class = torch.argmax(prediction, dim=1).item()
        response = {
            "probabilities": probabilities,
            "predicted_class": top_class
        }
        return json.dumps(response)

    raise ValueError(f"Unsupported content type: {content_type}")
