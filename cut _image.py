from PIL import Image
import os

def split_image(image_path, output_dir, tile_width=64, tile_height=64):
    """
    Splits an image into smaller tiles and saves each tile independently.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output images.
        tile_width (int): Width of each tile (default is 30).
        tile_height (int): Height of each tile (default is 30).
    """
    # Open the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Counter for naming the images
    count = 0

    # Loop through the image and extract tiles
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            # Define the box for cropping
            box = (x, y, x + tile_width, y + tile_height)
            tile = image.crop(box)

            # Save each tile
            tile_name = os.path.join(output_dir, f"tile_{count}.png")
            tile.save(tile_name)
            count += 1

    print(f"Successfully split the image into {count} tiles and saved them in '{output_dir}'")

# Example usage
image_path = "GTSRB/Test/fake_samples_epoch_9_step_0.png"  # Replace with your image file path
output_dir = "GTSRB/Test"  # Directory to save individual tiles
split_image(image_path, output_dir)
