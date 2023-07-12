import json
import matplotlib.pyplot as plt
import numpy as np
from sagemaker.predictor import Predictor
 

# Training data for different models had different image sizes and it is often observed that the model performs best when the generated image
# has dimensions same as the training data dimension. For dimensions not matching the default dimensions, it may result in a black image.
# Stable Diffusion v1-4 was trained on 512x512 images and Stable Diffusion v2 was trained on 768x768 images.




def handler(event, context):
    payload = {
    "prompt": "A colorful photo of a castle in the middle of a forest with trees",
    "width": 400,
    "height": 400,
    "num_images_per_prompt": 1,
    "num_inference_steps": 100,
    "guidance_scale": 7.5,
    }

    endpoint = 'jumpstart-dft-stabilityai-stable-diffusion-v2'
    predictor = Predictor(endpoint)
    # queryOutput = query(predictor, "astronaut on a horse")
    queryOutput = query_endpoint_with_json_payload(predictor, payload)
    # generated_images, prompt = parse_response(queryOutput)
    generated_images, prompt = parse_response_multiple_images(queryOutput)
    
    # image = display_img_and_prompt(generated_images, prompt)
    for img in generated_images:
        display_img_and_prompt(img, prompt)

    return {"statusCode": 200, "body": 'OK'}


def query(model_predictor, text):
    """Query the model predictor."""

    encoded_text = text.encode("utf-8")

    query_response = model_predictor.predict(
        encoded_text,
        {
            "ContentType": "application/x-text",
            "Accept": "application/json",
        },
    )
    return query_response

def query_endpoint_with_json_payload(model_predictor, payload):
    """Query the model predictor with json payload."""
    encoded_payload = json.dumps(payload).encode("utf-8")
    query_response = model_predictor.predict(
        encoded_payload,
        {
            "ContentType": "application/json",
            "Accept": "application/json",
        },
        )
    return query_response

def parse_response_multiple_images(query_response):
    """Parse response and return generated image and the prompt"""
    response_dict = json.loads(query_response)
    return response_dict["generated_images"], response_dict["prompt"]



def parse_response(query_response):
    """Parse response and return generated image and the prompt"""

    response_dict = json.loads(query_response)
    return response_dict["generated_image"], response_dict["prompt"]


def display_img_and_prompt(img, prmpt):
    """Display hallucinated image."""
    plt.figure(figsize=(12, 12))
    plt.imshow(np.array(img))
    plt.axis("off")
    plt.title(prmpt)
    plt.show()
    plt.savefig("saved_image.jpg")