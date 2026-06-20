import cv2
import numpy as np
from scipy import signal

""" OPERATIONS """
# Shift operation function
def operation_shift(img: np.ndarray , offset: int, direction: chr) -> np.ndarray:
    """ Shifts images to one of the four directions for a given number of pixels.

    Args:
        img (np.ndarray): Input image
        offset (int): Number of pixels to shift
        direction (chr): Which direction to shift
            's' -> South
            'w' -> West
            'n' -> North
            'e' -> East

    Returns:
        img_ (np.ndarray): Output image
    """

    if offset < 0:
        raise ValueError("offset must be non-negative")

    if direction == 's':
        img_ = np.roll(img, shift=offset, axis=0)

    elif direction == 'w':
        img_ = np.roll(img, shift=-offset, axis=1)

    elif direction == 'n':
        img_ = np.roll(img, shift=-offset, axis=0)

    elif direction == 'e':
        img_ = np.roll(img, shift=offset, axis=1)

    else:
        raise Exception("Invalid axis in shift operation. Must be 'n', 'w', 's', or 'e'.")

    return img_

# Flip operation function
def operation_flip(img: np.ndarray, axis: chr) -> np.ndarray:
    """Flips the image in an axis

    Args:
        img (np.ndarray): Image input.
        direction (chr): Specify vertical or horizontal axis.

    Returns:
        img_ (np.ndarray): Output image
    """

    if axis == 'v':
        img_ = np.flip(img, 0)
    elif axis == 'h':
        img_ = np.flip(img, 1)
    else:
        raise Exception("Invalid axis. Must be 'v' or 'h'.")

    return img_

# Inverse operation function
def operation_inverse(img: np.ndarray) -> np.ndarray:
    """Returns the inverse of an image

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Inversed image
    """
    # This really takes the bitwise-NOT of the image!
    # But we can still use np.invert() or np.bitwise_not()
    img_ = ~img

    return img_

# Return image to grayscale
def operation_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts image to grayscale

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Grayscale image
    """    
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_

# Return the selected channel of the image either in color or grayscale 
def operation_channels(img: np.ndarray, channel: chr, colored: bool) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Define the channels
    blue, green, red = cv2.split(img)

    # Define channel having all zeros
    zeros = np.zeros(blue.shape, np.uint8)

    # merge zeros to make BGR image
    if channel == 'r':
        if colored:
            img_ = cv2.merge([zeros, zeros, red])
        else:
            img_ = red
    
    elif channel == 'b':
        if colored:
            img_ = cv2.merge([blue, zeros, zeros])
        else:
            img_ = blue

    elif channel == 'g':
        if colored:
            img_ = cv2.merge([zeros, green, zeros])
        else:
            img_ = green

    else: 
        raise Exception("Invalid channel. Must be 'r', 'g', or 'b'.")

    return img_

def operation_gaussian_blur(img: np.ndarray, amplitude_kernel: tuple[int, int]) -> np.ndarray:
    # Tuple elements must be unsigned odd integers
    # Kernel indexes can indicate directions. For example:
    # (3, 13) -> Vertical Blur
    # (13, 3) -> Horizontal Blur
    # (13, 13) -> Equal Blur
    if len(amplitude_kernel) != 2:
        raise ValueError("amplitude_kernel must contain 2 integers")
    if any((not isinstance(k, int) or k <= 0 or k % 2 == 0) for k in amplitude_kernel):
        raise ValueError("Kernel values must be positive odd integers")

    img_ = cv2.GaussianBlur(img, amplitude_kernel, 0)
    return img_

def operation_laplacian_sharpener(img: np.ndarray, 
                                  amplitude: float = 1,
                                  kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])) -> np.ndarray:
    # An example kernel:
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel = np.array(kernel, dtype=np.float32, copy=True)
    kernel *= amplitude
    
    img_ = cv2.filter2D(img, -1, kernel)
    return img_

def operation_convolve(img: np.ndarray, kernel: np.ndarray):
    grad = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    
    img_ = (np.absolute(grad), np.angle(grad))

    return img_


def operation_detect_edges(img: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Detects edges using Canny."""
    gray = operation_grayscale(img) if img.ndim == 3 else img
    return cv2.Canny(gray, threshold1, threshold2)


def operation_first_derivative(img: np.ndarray, direction: str = "both") -> np.ndarray:
    """Computes first-order derivatives using Sobel operator.
    
    Args:
        img (np.ndarray): Input image (BGR or grayscale)
        direction (str): Derivative direction - "x" (horizontal), "y" (vertical), or "both"
    
    Returns:
        np.ndarray: Image showing first-order derivatives
    """
    gray = operation_grayscale(img) if img.ndim == 3 else img
    
    if direction == "x":
        # Horizontal derivative
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        img_ = cv2.convertScaleAbs(sobelx)
    elif direction == "y":
        # Vertical derivative
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        img_ = cv2.convertScaleAbs(sobely)
    else:  # "both"
        # Combined magnitude of both derivatives
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        img_ = cv2.convertScaleAbs(magnitude)
    
    return img_


def operation_second_derivative(img: np.ndarray) -> np.ndarray:
    """Computes second-order derivative using Laplacian operator.
    
    Highlights edges and corners where curvature changes significantly.
    
    Args:
        img (np.ndarray): Input image (BGR or grayscale)
    
    Returns:
        np.ndarray: Image showing second-order derivatives (Laplacian)
    """
    gray = operation_grayscale(img) if img.ndim == 3 else img
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    img_ = cv2.convertScaleAbs(laplacian)
    
    return img_


def operation_fourier_transform(img: np.ndarray, return_spectrum: bool = False):
    """Computes 2D Fast Fourier Transform preserving color information.
    
    Shows frequency components of the image in log scale.
    For color images, performs FFT on each channel separately.
    For grayscale, performs standard 2D FFT.
    
    Args:
        img (np.ndarray): Input image (BGR or grayscale)
        return_spectrum (bool): If True, also returns spectrum data for reconstruction
    
    Returns:
        If return_spectrum=False:
            np.ndarray: Magnitude spectrum for display (grayscale uint8)
        If return_spectrum=True:
            tuple: (magnitude_spectrum_display, spectrum_data_dict)
                spectrum_data_dict contains all info needed for inverse transform
    """
    if img.ndim == 3:  # Color image (BGR)
        # Perform FFT on each channel separately to preserve color
        channels = cv2.split(img)  # [B, G, R]
        spectra = []
        display_magnitude = None
        
        for i, channel in enumerate(channels):
            f_transform = np.fft.fft2(channel.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            spectra.append(f_shift)
            
            # Use first channel's magnitude for display
            if i == 0:
                magnitude = np.abs(f_shift)
                magnitude_log = np.log1p(magnitude)
                magnitude_normalized = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
                display_magnitude = np.uint8(magnitude_normalized)
        
        spectrum_data = {
            'type': 'color',
            'spectra': spectra,  # [B_spectrum, G_spectrum, R_spectrum]
            'shape': img.shape
        }
        
        if return_spectrum:
            return display_magnitude, spectrum_data
        return display_magnitude
    
    else:  # Grayscale image
        f_transform = np.fft.fft2(img.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        
        magnitude_spectrum = np.abs(f_shift)
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)
        magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX)
        img_ = np.uint8(magnitude_spectrum_normalized)
        
        spectrum_data = {
            'type': 'grayscale',
            'spectrum': f_shift,
            'shape': img.shape
        }
        
        if return_spectrum:
            return img_, spectrum_data
        return img_


def operation_inverse_fourier(spectrum_data) -> np.ndarray:
    """Computes inverse FFT from spectrum data, preserving color if it was color.

    Args:
        spectrum_data (dict): Spectrum data returned by operation_fourier_transform.
            Must contain 'type' key ('color' or 'grayscale').

    Returns:
        np.ndarray: Reconstructed spatial-domain image (BGR or grayscale uint8).
                   Returns original color format if input was color.
    """
    if spectrum_data is None:
        raise ValueError("spectrum_data cannot be None")
    
    if not isinstance(spectrum_data, dict):
        raise ValueError("spectrum_data must be a dictionary from operation_fourier_transform")
    
    spectrum_type = spectrum_data.get('type')
    
    if spectrum_type == 'color':
        # Reconstruct each channel separately
        spectra = spectrum_data.get('spectra')
        if not spectra or len(spectra) != 3:
            raise ValueError("Color spectrum data must contain 3 channel spectra")
        
        channels = []
        for f_shift in spectra:
            inv_shift = np.fft.ifftshift(f_shift)
            reconstructed = np.fft.ifft2(inv_shift)
            reconstructed_real = np.real(reconstructed)
            reconstructed_real = np.clip(reconstructed_real, 0, 255)
            channels.append(reconstructed_real.astype(np.uint8))
        
        # Merge back to BGR color image
        result = cv2.merge(channels)  # Merges as [B, G, R]
        return result
    
    elif spectrum_type == 'grayscale':
        # Reconstruct from single channel FFT
        f_shift = spectrum_data.get('spectrum')
        if f_shift is None:
            raise ValueError("Grayscale spectrum data must contain 'spectrum' key")
        
        inv_shift = np.fft.ifftshift(f_shift)
        reconstructed = np.fft.ifft2(inv_shift)
        reconstructed_real = np.real(reconstructed)
        reconstructed_real = np.clip(reconstructed_real, 0, 255)
        return reconstructed_real.astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown spectrum type: {spectrum_type}. Must be 'color' or 'grayscale'")


if __name__ == "__main__":
    demo_img = cv2.resize(cv2.imread("lenna.jpeg"), (0, 0), fx=0.5, fy=0.5)
    if demo_img is None:
        raise FileNotFoundError("lenna.jpeg was not found in the working directory")

    # Resize the image to make processing faster (optional)
    demo_img = cv2.resize(demo_img, (0, 0), fx=0.5, fy=0.5)

    transformed = operation_laplacian_sharpener(demo_img, 1)
    cv2.imshow("transformed_lenna", transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
shift+

horizontal flip+
vertical flip+

inverse+

grayscale+
RGB channel+

blur+
sharpen+

convolute+

derive+
integrate

fourier+
inverse fourier+
"""