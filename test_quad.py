import pytest
import cv2
import numpy as np
import sys
sys.path.append('build')
import quad  # Import your pybind11 module

@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_calculate_histogram_cv(sample_image):
    # Test the calculate_histogram_cv function
    hist = quad.calculate_histogram_cv(sample_image)
    assert len(hist) == 3 * 256  # 3 channels with 256 bins each

def test_cropImage(sample_image):
    # Test the cropImage function
    box = (10, 10, 50, 50)
    cropped_image = quad.cropImage(sample_image, box)
    assert cropped_image.shape == (40, 40, 3)

def test_weighted_average():
    # Test the weighted_average function
    hist = [0.2, 0.4, 0.6, 0.8]
    result = quad.weighted_average(hist)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)

def test_color_from_histogram():
    # Test the color_from_histogram function
    hist = [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.4]
    result = quad.color_from_histogram(hist)
    assert len(result) == 2
    assert isinstance(result[0], tuple)
    assert isinstance(result[1], float)

# Add more tests as needed
