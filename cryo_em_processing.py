import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

class CryoEMProcessor:
    def __init__(self, reference_img):
        """Initialize with reference image (img 100)"""
        self.reference = reference_img
        self.current_average = reference_img.copy()
        self.aligned_images = [reference_img]
        
    def cross_correlate(self, img1, img2):
        """
        Compute cross-correlation using FFT
        """
        
        # Compute FFTs
        fft1 = fft.fft2(img1)
        fft2 = fft.fft2(img2)
        
        # Compute cross-correlation
        correlation = fft.ifft2(fft1 * np.conj(fft2))
        correlation = fft.fftshift(correlation)
        
        return np.abs(correlation)
    
    def find_shift(self, img):
        """
        Find optimal x,y shifts using cross-correlation
        """
        correlation = self.cross_correlate(self.current_average, img)
        
        # Find peak correlation position
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Convert to shifts
        center_y, center_x = correlation.shape[0]//2, correlation.shape[1]//2
        shift_y = y - center_y
        shift_x = x - center_x
        
        # Get correlation value at peak
        max_correlation = np.max(correlation)
        
        return shift_x, shift_y, max_correlation
    
    def apply_shift(self, img, shift_x, shift_y):
        """Apply translation using np.roll"""
        shifted = np.roll(img, shift_y, axis=0)
        shifted = np.roll(shifted, shift_x, axis=1)
        return shifted
    
    def update_average(self, new_img, img_count):
        """Update running weighted average"""
        # weight = 1.0 / img_count
        self.current_average = (img_count / (img_count + 1))*self.current_average + (1 / (img_count + 1))*new_img
    
    def align_single_image(self, img, debug=False):
        """
        Align a single image and return correlation value
        """
        # Find optimal shift
        shift_x, shift_y, correlation = self.find_shift(img)
        
        if debug:
            print(f"Shifts found: x={shift_x}, y={shift_y}, correlation={correlation:.3f}")
            
        # Apply shift
        aligned_img = self.apply_shift(img, shift_x, shift_y)
        
        return aligned_img, correlation
    
    def process_batch(self, images, correlation_threshold=0.8, show_progress=True):
        """Process a batch of images"""
        accepted_count = 0
        rejected_count = 0
        
        for i, img in enumerate(images, 1):
            # Align image
            aligned_img, correlation = self.align_single_image(img, debug=(i % 50 == 0))
            
            # If correlation is good enough, update average
            if correlation > correlation_threshold:
                self.update_average(aligned_img, len(self.aligned_images))
                self.aligned_images.append(aligned_img)
                accepted_count += 1
            else:
                rejected_count += 1
            
            # Show progress every 50 images
            if show_progress and i % 50 == 0:
                print(f"Processed {i} images. Accepted: {accepted_count}, Rejected: {rejected_count}")
                plt.figure(figsize=(10, 4))
                
                plt.subplot(121)
                plt.imshow(img, cmap='gray')
                plt.title('Original Image')
                
                plt.subplot(122)
                plt.imshow(self.current_average, cmap='gray')
                plt.title(f'Current Average (n={len(self.aligned_images)})')
                
                plt.show()
        
        print(f"\nFinal stats:")
        print(f"Total processed: {len(images)}")
        print(f"Accepted: {accepted_count}")
        print(f"Rejected: {rejected_count}")
        
        return self.current_average