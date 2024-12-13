import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cryo_em_processing import CryoEMProcessor

# Visualization control flags
SHOW_REFERENCE = False
SHOW_PROGRESS = False  # Shows progress every 50 images during processing
SHOW_SET1 = True
SHOW_SET2 = True
SHOW_FINAL = True

def load_images(directory):
    """Load all .tif images from directory"""
    images = []
    filenames = sorted([f for f in os.listdir(directory) if f.endswith('.tif')])
    
    print(f"Found {len(filenames)} images")
    
    for i, filename in enumerate(filenames):
        if i % 50 == 0:
            print(f"Loading image {i}/{len(filenames)}")
        path = os.path.join(directory, filename)
        img = np.array(Image.open(path))
        images.append(img)
    
    return images

def process_images():
    # Load all images
    print("Loading images...")
    all_images = load_images('dip-project-imgs')
    
    # Get reference image (image 100)
    reference_img = all_images[99]
    
    # Show reference image
    if SHOW_REFERENCE:
        plt.figure(figsize=(8, 8))
        plt.imshow(reference_img, cmap='gray')
        plt.title('Reference Image (100)')
        plt.show()
    
    # Split into two sets
    set1 = all_images[:250]
    set2 = all_images[250:]
    
    # Process first set
    print("\nProcessing first set of 250 images...")
    processor1 = CryoEMProcessor(reference_img)
    result1 = processor1.process_batch(set1, show_progress=SHOW_PROGRESS)
    
    if SHOW_SET1:
        # Save and show result from first set
        plt.figure(figsize=(8, 8))
        plt.imshow(result1, cmap='gray')
        plt.title('Result from First Set')
        plt.savefig('result_set1.png')
        plt.show()
    
    # Process second set
    print("\nProcessing second set of 250 images...")
    processor2 = CryoEMProcessor(reference_img)
    result2 = processor2.process_batch(set2, show_progress=SHOW_PROGRESS)
    
    if SHOW_SET2:
        # Save and show result from second set
        plt.figure(figsize=(8, 8))
        plt.imshow(result2, cmap='gray')
        plt.title('Result from Second Set')
        plt.savefig('result_set2.png')
        plt.show()
    
    # Final combination
    print("\nPerforming final combination...")
    final_processor = CryoEMProcessor(result1)
    final_result = final_processor.process_batch([result2], show_progress=SHOW_PROGRESS)
    
    if SHOW_FINAL:
        # Save and show final result
        plt.figure(figsize=(8, 8))
        plt.imshow(final_result, cmap='gray')
        plt.title('Final Combined Result')
        plt.savefig('final_result.png')
        plt.show()
    
    return final_result

if __name__ == "__main__":
    try:
        final_image = process_images()
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")