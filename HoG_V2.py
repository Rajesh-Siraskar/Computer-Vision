
# coding: utf-8

# 07-Jan-2019: Version 2: Five 'for' loops are reduced to 'two'
#                         Three for loops eliminated: (1) Inner num_bins (2) Pixel x and (3) pixel y locations 


# Import standard modules
import cv2
import numpy as np

## Function to compute histogram of gradients for an image

## Parameters:
# Cells: A unit block of pixels
#    cell_size = size of cell in nxn pixels
# Block cells: n cells
#    block_cells = multiplier n i.e. n cells HxW 
# Block step: Cells by which steps are made - scanning bounding box left to right and top to bottom
#    step_cells: n cells
# Number of 'directional' (bins of direction i.e. angle of the gradient vectors)
#   20 deg. bins => 180 degs./20 = 9 bins

def HoG_V2 (image, cell_size = (8, 8), block_cells = 2, step_cells = 1, num_bins = 9, apply_filter = None, filter_param = None):

    # Resize image to recommended size 64×128 [(Dalal N., 2006) Table 4.2. Key HOG parameters :pg 47]
    
    # Size of bounding box     
    image_size = (image.shape[0], image.shape[1])
    
    # Warning: cv2.resize in cols x rows 
    recommended_size = (64, 128)
    image = cv2.resize(image, recommended_size)
    
    # Block size nxn cells. n = block_cells
    block_size =  (block_cells*cell_size[0], block_cells*cell_size[1])

    # Block stride - can move say one cell at a time, scanning left to right and top to bottom
    block_stride = step_cells*cell_size[0]

    ### Compute Histogram of Gradients
    # Algorithm: Klette 'Concise Computer Vision'. Pg.382
    # Refinements based on the original thesis (Dalal N., 2006)
    
    # Step 1.: Pre-processing
    #          [a] Intensity normalization
    #          [b] Smoothing filter
    
    # Step 1.a: Normalize with highest value of intensity
    #    Simplest form of normalization used
    #    Thesis suggests 'square root gamma compression'
    
    image = np.float32(image)/255.0
    
    # Step 1.b: Bilateral Filtering
    #    Normal smoothing including Gaussian blur do remove noise but affect the edges too.
    #    For pedestrian detection it might be beneficial to preserve the edges - hence we use a bilateral filter instead
    #    As stated in OpenCV-Python Tutorials (Mordvintsev A. and Abid K., 2014), cv2.bilateralFilter() is highly effective 
    #      at noise removal while still preserving edges at the cost of the operation being slower compared to other filters
    #      https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    #    Recommended values ref https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    # 
    #    The original thesis (Dalal N., 2006: pg. 37 - 4.3.2 Gradient Computation) suggests
    #      recommends sigma=0 i.e. NO smoothing
    #
    #    However tests showed that the edges were more prominient with use of the bi-lateral filter
    
    if (apply_filter == 'GAUSSIAN'): 
        if (filter_param is None): filter_param = 5
        # All three parameters (ksize.width, ksize.height) and sigma assumed to take the value filter_param
        image = cv2.GaussianBlur(image, (filter_param, filter_param), filter_param)
        
    if (apply_filter == 'BILATERAL'): 
        if (filter_param is None): filter_param = 80
        image = cv2.bilateralFilter(image, 7, filter_param, filter_param)
    
    # Step 2.a.: Calculate gradients for each pixel
    #   Use Sobel filter with a kernel size 1 pixel x 1 pixel
    #   Ref.: (Dalal N., 2006: pg. 37 - 4.3.2 Gradient Computation) suggests Sobel operator [−1, 0, 1]
    #   Use un-signed gradients pg. 48. '4.6 Experiments on Other Classes'
    
    grad_x = abs(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1))
    grad_y = abs(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1))


    # Step 2.b.: Compute the magnitude and angles of the gradient vectors 
    Im, Ia = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    

    # Step.3: Spatial binning
    # Group pixels in non-overlapping cells (e.g. 8×8 pixels)
    # Use maps Im and Ia to accumulate magnitude values into direction bins 
    #  (e.g., nine bins for intervals of 20◦ each, for covering a full 180◦ range) 
    #  to obtain a voting vector (e.g. of length 9) for each cell
    
    # 1 histogram per cell with 9 bins
    # Create a collection of bins of num_bins length
    rows = int(image.shape[0]/cell_size[0])
    cols = int(image.shape[1]/cell_size[1])
    Im_bins =  np.zeros((rows, cols, num_bins))
    
    # Step 3.a.: Create bin centers (e.g. 0, 20, 40...160 deg., if num_bins=9)
    angle_slices = int(180.0/num_bins)
    Ia_bin_centers = [angle for angle in range(0, 180, angle_slices)]

 
    ## Traverse the image starting at top, then left to right, then next row...

    # Keep track of cells
    n_cell = 0
    n_row = 0
    n_col = 0

    # Histogram bin band-width
    b = Ia_bin_centers[1] - Ia_bin_centers[0]

    # Move T-B
    n_row = 0
    for y in range(0, image.shape[0], block_stride):
        # Move L-R, intialize column counter
        n_col = 0
        for x in range(0, image.shape[1], block_stride):
            Im_bins = _HoG_histogram_binning (Im, Ia, Im_bins, Ia_bin_centers, b, n_row, n_col, x, y, cell_size, num_bins)
            # Complete cell covered, move to next cell 
            n_col += 1
        # 1 row of cells L-R covered, proceed to next row
        n_row+=1
    # All rows covered L-R, complete image processed

    # Create a feature vector
    hog_features = Im_bins.ravel().reshape(-1, 1)
                    
    return (hog_features)

### Process Im and bin it
def _HoG_process_bin (Im, Ia, Im_bins, Ia_bin_centers, b, n_row, n_col, px_row, px_col, n_bin, num_bins):
    
    if (n_bin > -1):
        # Allow if we are at last-but-one bin, but if last bin then split share with 0 deg. (1st) bin:
        next_bin = (n_bin+1) if (n_bin+1 < num_bins) else 0

        Im_bins[n_row][n_col][n_bin] += Im[px_row,px_col] * (1 - (Ia[px_row,px_col] - Ia_bin_centers[n_bin])/b)
        Im_bins[n_row][n_col][next_bin] += Im[px_row,px_col] * (Ia[px_row,px_col] - Ia_bin_centers[n_bin])/b
    
    return(Im_bins)
   
### Process each pixel and if the Ia is within the bin-centers' range then call  _HoG_process_bin 
def _HoG_process_pixel (Im, Ia, Im_bins, Ia_bin_centers, b, n_row, n_col,  x, y, px_row, px_col, num_bins):
    
    
    [_HoG_process_bin (Im, Ia, Im_bins, Ia_bin_centers, b, n_row, n_col, px_row, px_col, n_bin, num_bins) 
     if (Ia[px_row, px_col] >= Ia_bin_centers[n_bin] and Ia[px_row, px_col] < Ia_bin_centers[n_bin+1]) else None 
     for n_bin in range(9)] 

### Access each pixel in the cell and call _HoG_process_pixel for each pixel
### Binning logic: Dalal N., 2006: Appendix D
# Place the pixel magnitude Im in the appropriate bin
# x1 <= x < x2
# h(x1) <- h(x1) + w * [1 - (x-x1)/b]
# h(x2) <- h(x1) + w * (x-x1)/b
# For this implementation it is assumed that w = Im at that pixel

def _HoG_histogram_binning (Im, Ia, Im_bins, Ia_bin_centers, b, n_row, n_col, x, y, cell_size, num_bins):
    
    [_HoG_process_pixel(Im, Ia, Im_bins, Ia_bin_centers, b, n_row, n_col,  x, y, px_row, px_col, num_bins) 
     for px_row in range(y, y+cell_size[1]) for px_col in range(x, x+cell_size[0])]
   
    return (Im_bins)
