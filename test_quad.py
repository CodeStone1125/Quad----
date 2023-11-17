import pytest
from PIL import Image
import cv2
import numpy as np
import sys
sys.path.append('build')
import quad  # Import your pybind11 module

@pytest.fixture
def sample_image():
    # Load a sample image for testing
    image_path = "assests/star.jpg"
    return cv2.imread(image_path)


# def test_calculate_histogram_cv(sample_image):
#     # Test the calculate_histogram_cv function
#     hist = quad.calculate_histogram_cv(sample_image)
#     assert len(hist) == 3 * 256  # 3 channels with 256 bins each

# def test_cropImage():

#     box = (10, 10, 50, 50)
#     black_image = np.zeros((100, 100, 3), dtype=np.uint8)
#     # Load the sample image for testing
#     im = Image.open("assests/star.jpg").convert('RGB')
#     pil_cropped_im = im.crop(box)

#     # Convert PIL image to NumPy array
#     pil_cropped_im = np.array(pil_cropped_im)

#     # python CV2 to get RGB image
#     im_cv2 = cv2.imread("assests/star.jpg")
#     # Convert NumPy array to cv::Mat
#     im_cv2_rgb = cv2.cvtColor(im_cv2, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR, so convert RGB to BGR
#     im_cv2_rgb = np.array(im_cv2_rgb)
#     print(type(im_cv2_rgb))
#     print(type((10, 10, 50, 50)))
#     quad_cropped_im = quad.cropImage(black_image,(10, 10, 50, 50))
    

#     # Assert that the cropped images are equal
#     assert np.array_equal(quad_cropped_im, pil_cropped_im)

def test_cropImage():
    box = (10, 10, 50, 50)

    # Load the sample image for testing
    im = Image.open("assests/star.jpg").convert('RGB')
    pil_cropped_im = im.crop(box)

    # Convert PIL image to NumPy array
    pil_cropped_im = np.array(pil_cropped_im)

    # python CV2 to get RGB image
    im_cv2 = cv2.imread("assests/star.jpg")
    # Convert NumPy array to cv::Mat
    im_cv2_rgb = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)
    im_cv2_mat = cv2.Mat(im_cv2_rgb)  # Assuming cv2Mat is your binding for cv::Mat

    print(type(im_cv2_mat))
    print(type(box))
    # Get the cropped image using the cropImage function
    quad_cropped_im = quad.cropImage(im_cv2_mat, (10, 10, 50, 50))

    # Assert that the cropped images are equal
    assert np.array_equal(quad_cropped_im, pil_cropped_im)



def test_weighted_average():
    # Test the weighted_average function
    hist = [0.2, 0.4, 0.6, 0.8]
    result = quad.weighted_average(hist)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)

# def test_color_from_histogram():
#     # Test the color_from_histogram function
#     hist = [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.4]
#     result = quad.color_from_histogram(hist)
#     assert len(result) == 2
#     assert isinstance(result[0], tuple)
#     assert isinstance(result[1], float)

# Add more tests as needed
