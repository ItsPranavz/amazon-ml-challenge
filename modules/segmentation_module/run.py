import torch
import warnings
import os
import glob
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

warnings.filterwarnings('ignore')

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Set the input folder (where the images are located) and output folder
input_dir = '../../data/images/'
output_dir = '../../data/segments/'

# Load the models (adjust based on CUDA availability)
refine_net = load_refinenet_model(cuda=use_cuda)
craft_net = load_craftnet_model(cuda=use_cuda)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image paths from the folder (adjust extensions as needed)
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

# Process each image
for image_path in image_paths:
    # Read the image
    image = read_image(image_path)

    # Perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=use_cuda,  # Use GPU if available
        long_size=1280
    )

    # Get image file name without extension to use in output file names
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create a subfolder for each image's segments (optional)
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Export detected text regions for each image
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=image_output_dir,
        rectify=True
    )

    # Export extra results (heatmaps, detection points)
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=image_output_dir
    )

    print(f'Processed: {image_path}')

# Unload models from GPU
empty_cuda_cache()

print('All images processed successfully.')