import cv2
import numpy as np
from PIL import Image

def noiseless_mask(mask):
    """
    Cleans a binary mask by removing noise and small artifacts while preserving main objects.
    
    Args:
        mask: Binary input mask where white (255 or 1) represents objects
              and black (0) represents background
              
    Returns:
        cleaned_mask: A binary mask with small noise elements removed
    """
    
    cleaned_mask = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat_result = np.zeros_like(mask)
    tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
    tophat_result = np.maximum(tophat_result, tophat)
    enhanced = cv2.add(mask, tophat_result)
    dist_transform = cv2.distanceTransform(enhanced, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
    areas = np.bincount(labels.flatten())
    lut = np.zeros(num_labels, dtype=np.uint8)
    lut[1:] = np.where(areas[1:] >= 20, 255, 0)
    cleaned_mask = lut[labels]
    
    return cleaned_mask
    
def adaptive_dullrazor(img_path, lowbound=20, showimgs=True, inpaintmat=3):
    """
    Removes hair-like structures from images using an adaptive DullRazor approach.
    
    The function adjusts the morphological filter size based on image complexity
    measured by edge density and variance, then performs hair detection and removal
    through morphological operations and inpainting.
    
    Args:
        img_path: Path to the input image
        lowbound: Threshold value for detecting dark structures (default=20)
        showimgs: Boolean flag to display intermediate results (default=True)
        inpaintmat: Size of neighborhood for inpainting (default=3)
        
    Returns:
        img_final: The processed image with hair-like structures removed
        mask: Binary mask showing detected hair-like structures
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    filterstruc = max(3, min(12, int(10 * (1 - edge_density) + 3)))
    filterSize = (filterstruc, filterstruc)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, initial_mask = cv2.threshold(blackhat, lowbound, 255, cv2.THRESH_BINARY)
    mask=noiseless_mask(initial_mask)
    img_final = cv2.inpaint(img, mask, inpaintmat, cv2.INPAINT_TELEA)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    return img_final,initial_mask,mask


def smoothen_rounded_border(image):
    """
    Uses a radial gradient-based approach to extend colors from the circle to the black area.
    Only processes images where:
    1. The circle is centered and covers at least 25% of the image
    2. The area outside the circle has at least 60% of pixels with values less than 10 (nearly black)
    
    Args:
        image: Input RGB image with a circular content area and dark background
        
    Returns:
        result: Image with smoothened border transition or original image if criteria not met
    """
    
    #print(image.dtype)
    result = image.copy()
    img_height, img_width = image.shape[:2]
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    img_area = img_height * img_width
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=40, maxRadius=200
    )
    if circles is None:
        return image

    circles = np.uint16(np.around(circles))
    if len(circles[0]) > 0:
        center_x, center_y, radius = circles[0][0]
        circle_area = np.pi * (radius ** 2)
        area_percentage = (circle_area / img_area) * 100
        center_x_diff = abs(center_x - img_center_x)
        center_y_diff = abs(center_y - img_center_y)
        x_threshold = img_width * 0.1
        y_threshold = img_height * 0.1
        
        is_centered = center_x_diff < x_threshold and center_y_diff < y_threshold
        covers_enough = area_percentage >= 25
        circle_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
        circle_mask = cv2.GaussianBlur(circle_mask, (15,15),0)
        # Create outside circle mask - white (255) outside the circle, black (0) inside
        outside_mask = cv2.bitwise_not(circle_mask)
        outside_indices = np.where(outside_mask > 0)
        background_pixels = gray[outside_indices]
        
        # Check if at least 40% of the background pixels have values less than 10
        dark_pixels_ratio = np.sum(background_pixels < 10) / len(background_pixels) if len(background_pixels) > 0 else 0
        has_dark_background = dark_pixels_ratio >= 0.4
       
        # Only process if the circle meets all criteria
        if is_centered and covers_enough and has_dark_background:

            y_coords, x_coords = np.ogrid[:img_height, :img_width]
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            angles = np.arctan2(y_coords - center_y, x_coords - center_x + 1e-10)
            fade_distance = radius * 3.5  # How far outside the circle to extend the gradient
            alpha = np.zeros((img_height, img_width), dtype=np.float32)
            
            # Get mask as boolean array
            outside_bool = outside_mask > 0
            outside_distances = dist_from_center[outside_bool]
            alpha[outside_bool] = 1.0 - ((outside_distances - radius) / fade_distance)
            alpha[alpha < 0] = 0  
            
            # Create output image
            result = image.copy()
            #plt.imshow(result)
            y_outside, x_outside = outside_indices
            
           
            for i in range(len(y_outside)):
                y = y_outside[i]
                x = x_outside[i]  
                # Get the angle of this pixel
                pixel_angle = angles[y, x]
                # Find the corresponding point inside the circle
                sample_radius = radius * 0.8  # Sample from inside circle to avoid edge effects
                sample_x = int(center_x + sample_radius * np.cos(pixel_angle))
                sample_y = int(center_y + sample_radius * np.sin(pixel_angle))         
                # Ensure sample point is within image bounds
                sample_x = max(0, min(sample_x, img_width - 1))
                sample_y = max(0, min(sample_y, img_height - 1))           
                # Get the sampled color
                sampled_color = image[sample_y, sample_x].astype(np.float32)        
                # Blend original with sampled color based on alpha
                weight = alpha[y, x]
                result[y, x] = (1.0 - weight) * result[y, x] + weight * sampled_color
            
            # Convert back to uint8    
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
            
        else:
            return image
    else:
        return image

def preprocessing(img_path):
    """
    Preprocess a skin lesion image by removing hair and smoothing  round borders.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        final: Processed image
    """
    img_final,initial_mask,mask = adaptive_dullrazor(img_path)
    final = smoothen_rounded_border(img_final)
    final= cv2.cvtColor(final, cv2.COLOR_RGB2BGR)    
    return final

