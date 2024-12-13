import matplotlib.pyplot as plt
import numpy as np
from cryo_em_processing import CryoEMProcessor

class ConvergenceTracker:
    def __init__(self):
        self.correlations = []
        self.acceptance_rates = []
        self.cumulative_accepted = []
        self.running_avg_correlations = []
        
    def add_data(self, correlation, accepted_count, total_processed):
        self.correlations.append(correlation)
        acceptance_rate = accepted_count / total_processed
        self.acceptance_rates.append(acceptance_rate)
        self.cumulative_accepted.append(accepted_count)
        
        # Calculate running average of correlations
        window_size = min(50, len(self.correlations))
        running_avg = np.mean(self.correlations[-window_size:])
        self.running_avg_correlations.append(running_avg)
    
    def plot_convergence(self, title_prefix=""):
        plt.figure(figsize=(15, 5))
        
        # Plot correlation values
        plt.subplot(131)
        plt.plot(self.correlations, 'b-', alpha=0.3, label='Raw Correlations')
        plt.plot(self.running_avg_correlations, 'r-', label='Running Average')
        plt.xlabel('Image Number')
        plt.ylabel('Correlation Value')
        plt.title(f'{title_prefix}Correlation Values Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot acceptance rate
        plt.subplot(132)
        plt.plot(self.acceptance_rates, 'g-')
        plt.xlabel('Image Number')
        plt.ylabel('Acceptance Rate')
        plt.title(f'{title_prefix}Acceptance Rate Over Time')
        plt.grid(True)
        
        # Plot cumulative accepted images
        plt.subplot(133)
        plt.plot(self.cumulative_accepted, 'r-')
        plt.xlabel('Image Number')
        plt.ylabel('Cumulative Accepted Images')
        plt.title(f'{title_prefix}Cumulative Accepted Images')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

class CryoEMProcessorWithTracking(CryoEMProcessor):
    def __init__(self, reference_img):
        super().__init__(reference_img)
        self.tracker = ConvergenceTracker()
        
    def process_batch(self, images, correlation_threshold=0.8, show_progress=True):
        accepted_count = 0
        rejected_count = 0
        
        for i, img in enumerate(images, 1):
            # Align image
            aligned_img, correlation = self.align_single_image(img, debug=(i % 50 == 0))
            
            # Track convergence data
            if correlation > correlation_threshold:
                accepted_count += 1
            else:
                rejected_count += 1
                
            self.tracker.add_data(correlation, accepted_count, i)
            
            # If correlation is good enough, update average
            if correlation > correlation_threshold:
                self.update_average(aligned_img, len(self.aligned_images))
                self.aligned_images.append(aligned_img)
            
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
        
        # Plot convergence at the end
        self.tracker.plot_convergence()
        
        print(f"\nFinal stats:")
        print(f"Total processed: {len(images)}")
        print(f"Accepted: {accepted_count}")
        print(f"Rejected: {rejected_count}")
        
        return self.current_average