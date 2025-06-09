import os
import random
from PIL import Image, ImageEnhance, ImageFilter

# Paths to your datasets
scene_dataset_path = "real_scenes"
traffic_sign_dataset_path = "GTSRB/Train"

output_path = "fake_scenes"

# Function to add a traffic sign to a scene image with realistic adjustments
def overlay_traffic_sign(scene_image, traffic_sign_dir_path):
    traffic_sign_image = random.choice([os.path.join(traffic_sign_dir_path, f)
                                        for f in os.listdir(traffic_sign_dir_path)
                                        if f.endswith(('.png', '.jpg', '.jpeg'))])
    # Load the images
    scene = Image.open(scene_image)
    traffic_sign = Image.open(traffic_sign_image).convert("RGBA")

    # Resize traffic sign to a realistic scale
    scale_factor = random.uniform(0.8, 0.8)  # Random scale for variation
    sign_width, sign_height = traffic_sign.size
    new_sign_size = (int(sign_width * scale_factor), int(sign_height * scale_factor))
    traffic_sign = traffic_sign.resize(new_sign_size, Image.LANCZOS)

    # Position the traffic sign at a random realistic location
    max_x = scene.width - traffic_sign.width
    max_y = int(scene.height * 0.6)  # Limit height for realistic sign placement
    position = (random.randint(0, max_x), random.randint(int(scene.height * 0.4), max_y))

    # Adjust brightness and contrast to match the scene
    enhancer = ImageEnhance.Brightness(traffic_sign)
    traffic_sign = enhancer.enhance(random.uniform(0.8, 1.2))  # Adjust brightness
    enhancer = ImageEnhance.Contrast(traffic_sign)
    traffic_sign = enhancer.enhance(random.uniform(0.8, 1.2))  # Adjust contrast

    # Apply blur if the sign is distant (optional for realism)
    if position[1] > scene.height * 0.5:  # More blur if farther in the background
        traffic_sign = traffic_sign.filter(ImageFilter.GaussianBlur(1.5))

    # Add a shadow for depth
    shadow = traffic_sign.copy().convert("RGBA")
    shadow = shadow.point(lambda p: p * 0)  # Make the shadow black
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))  # Blur shadow
    shadow_position = (position[0] + 5, position[1] + 5)  # Offset shadow

    # Overlay the shadow and traffic sign on the scene
    scene.paste(shadow, shadow_position, shadow)
    scene.paste(traffic_sign, position, traffic_sign)

    return scene

# Iterate over scene and traffic sign images
scene_images = os.listdir(scene_dataset_path)
traffic_sign_images = os.listdir(traffic_sign_dataset_path)

for i, scene_filename in enumerate(scene_images):
    # Select random traffic sign for each scene
    traffic_sign_filename = random.choice(traffic_sign_images)
    scene_image_path = os.path.join(scene_dataset_path, scene_filename)
    traffic_sign_image_path = os.path.join(traffic_sign_dataset_path, traffic_sign_filename)

    # Overlay traffic sign on the scene
    result_image = overlay_traffic_sign(scene_image_path, traffic_sign_image_path)

    # Save the resulting image
    result_filename = f"result_{i}.jpg"
    result_image.save(os.path.join(output_path, result_filename))

print("Traffic signs have been overlayed and saved to the output folder.")
