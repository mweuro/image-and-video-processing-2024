from .__init__ import *


#==========IMAGE SCALING==========#

def scale_img(img, scale_percent = 25):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_image


#==========SIFT BETWEEN 2 IMAGES==========#

def display_matcher(img1, img2, ct, et, n, d):
    sift = cv2.SIFT_create(nfeatures = n, contrastThreshold = ct, edgeThreshold = et)
    kp1, ds1 = sift.detectAndCompute(img1, mask = None)
    kp2, ds2 = sift.detectAndCompute(img2, mask = None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ds1, ds2, k = 2)
    good = []
    for m, n in matches:
        if m.distance < d * n.distance:
            good.append([m])
    matches_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    _, axs = plt.subplots(1, 1, figsize = (16, 10))
    axs.imshow(matches_img);
    axs.set_title('Distance constant = {}'.format(d))
    axs.axis('off')
    plt.show()
    
    
def interactive_matcher(img1, img2):
    slider_layout = Layout(width = '40%')
    ct_slider = FloatSlider(value = 0.08, min = 0, max = 1, step = 0.01, description = "contrastThreshold", layout = slider_layout)
    et_slider = FloatSlider(value = 4.2, min = 0, max = 50, step = 0.1, description = "edgeThreshold", layout = slider_layout)
    n_slider = IntSlider(value = 200, min = 0, max = 500, step = 1, description = "nfeatures", layout = slider_layout)
    d_slider = FloatSlider(value = 0.75, min = 0, max = 1, step = 0.01, description = "distance constant", layout = slider_layout)

    interact(display_matcher, img1 = fixed(img1), img2 = fixed(img2), ct = ct_slider, et = et_slider, n = n_slider, d = d_slider)


#==========IMG1 HOMOGRAPHY ONTO IMG2 SPACE==========#

def object_homography(img1, img2, ct = 0.04, et = 50, d = 0.75, method = cv2.RANSAC, ransac_thresh = 5.0):
    sift = cv2.SIFT_create(nfeatures = 0, contrastThreshold = ct, edgeThreshold = et)
    kp1, ds1 = sift.detectAndCompute(img1, None)
    kp2, ds2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
    matches = bf.knnMatch(ds1, ds2, k = 2)

    good_matches = []
    for m, n in matches:
        if m.distance < d * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    if method == cv2.RANSAC or method == cv2.RHO:
        H, _ = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
    else:
        H, _ = cv2.findHomography(src_pts, dst_pts, method)
        
    height, width, _ = img2.shape
    transformed_img = cv2.warpPerspective(img1, H, (width, height))
    
    _, axs = plt.subplots(1, 3, figsize = (15, 5))
    axs[0].imshow(img1);
    axs[0].set_title('Object image')
    axs[0].axis('off')
    axs[1].imshow(transformed_img);
    axs[1].set_title(f'Object transform')
    axs[1].axis('off')
    axs[2].imshow(img2);
    axs[2].set_title('Destination image')
    axs[2].axis('off')
    plt.show()
    
    
def interactive_homography(img1, img2):
    slider_layout = Layout(width = '40%')
    ct_slider = FloatSlider(value = 0.04, min = 0, max = 1, step = 0.01, description = "contrastThreshold", layout = slider_layout)
    et_slider = FloatSlider(value = 50, min = 0, max = 200, step = 0.1, description = "edgeThreshold", layout = slider_layout)
    d_slider = FloatSlider(value = 0.75, min = 0, max = 1, step = 0.01, description = "distance constant", layout = slider_layout)
    method_dropdown = Dropdown(
            options = [('Brak metody', 0), ('RANSAC', cv2.RANSAC), ('RHO', cv2.RHO)],
            value = cv2.RANSAC,
            description = 'Metoda:',
            layout = slider_layout
        )
    
    interact(object_homography, 
             img1 = fixed(img1), 
             img2 = fixed(img2), 
             ct = ct_slider, 
             et = et_slider, 
             d = d_slider, 
             method = method_dropdown)




def stitch_images(img1, img2, ct = 0.04, et = 50, d = 0.75, method = cv2.RANSAC, ransac_thresh = 5.0):
    img1 = img1.copy()
    img2 = img2.copy()
    
    sift = cv2.SIFT_create(nfeatures = 0, contrastThreshold = ct, edgeThreshold = et)
    kp1, ds1 = sift.detectAndCompute(img1, None)
    kp2, ds2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
    matches = bf.knnMatch(ds1, ds2, k = 2)

    good_matches = []
    for m, n in matches:
        if m.distance < d * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    if method == cv2.RANSAC or method == cv2.RHO:
        H, _ = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
    else:
        H, _ = cv2.findHomography(src_pts, dst_pts, method)
        
    
    height, width = img1.shape[:2]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    transformed_corners = cv2.perspectiveTransform(corners[None, :, :], H)

    min_x = min(transformed_corners[0][:, 0])
    max_x = max(transformed_corners[0][:, 0])
    min_y = min(transformed_corners[0][:, 1])
    max_y = max(transformed_corners[0][:, 1])
    new_width = int(max_x)
    new_height = int(max_y)
    
    transformed_img = cv2.warpPerspective(img1, H, (max(width, new_width), max(height, new_height)))
    final_img = transformed_img.copy()
    final_img[0:height, 0:width] = img2
    
    
    _, axs = plt.subplots(1, 3, figsize = (15, 7))
    axs[0].imshow(img2)
    axs[0].set_title('Base')
    axs[1].imshow(transformed_img)
    axs[1].set_title('Transformed offset')
    axs[2].imshow(final_img)
    axs[2].set_title('Stitched images')
    
    plt.show()