import cv2
import os

def clean_images(image_dir, output_dir, batch_size=32):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of all files in the image directory
    image_files = [filename for filename in os.listdir(image_dir) if filename.endswith((".jpg", ".png", ".jpeg"))]
    
    # Calculate the total number of batches
    num_batches = len(image_files) // batch_size
    
    # Loop over each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        
        # Load and process images in the current batch
        batch_images = []
        for filename in image_files[start_idx:end_idx]:
            # Construct the full file path
            filepath = os.path.join(image_dir, filename)
            
            # Load the image using OpenCV
            image = cv2.imread(filepath)
            
            # Resize the image to a fixed size (e.g., 300x300 pixels)
            resized_image = cv2.resize(image, (300, 300))
            
            # Append the resized image to the batch
            batch_images.append(resized_image)
        
        # Save the batch of cleaned images to the output directory
        batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx}")
        os.makedirs(batch_output_dir, exist_ok=True)
        for idx, image in enumerate(batch_images):
            output_filepath = os.path.join(batch_output_dir, f"image_{idx}.jpg")
            cv2.imwrite(output_filepath, image)

# Example usage
image_dir = r"D:\BANDICAM\Pneumonia\Train\PNEUMONIA"
output_dir = r"D:\BANDICAM\Pneumonia\Train\Pneumonia(clean)"
clean_images(image_dir, output_dir, batch_size=32)
