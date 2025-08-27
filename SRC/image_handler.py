"""
Image Handler - Menangani loading dan preprocessing gambar
"""
import logging
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from config import SUPPORTED_FORMATS

class ImageHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_image(self, image_path):
        """
        Validasi file gambar
        
        Args:
            image_path (Path): Path ke file gambar
            
        Returns:
            bool: True jika valid, False jika tidak
        """
        try:
            # Cek apakah file ada
            if not image_path.exists():
                self.logger.error(f"File tidak ditemukan: {image_path}")
                return False
            
            # Cek ekstensi file
            if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                self.logger.error(f"Format tidak didukung: {image_path.suffix}")
                self.logger.info(f"Format yang didukung: {', '.join(SUPPORTED_FORMATS)}")
                return False
            
            # Cek ukuran file (max 50MB)
            file_size = image_path.stat().st_size / (1024 * 1024)  # MB
            if file_size > 50:
                self.logger.error(f"File terlalu besar: {file_size:.1f}MB (max 50MB)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validasi gambar: {str(e)}")
            return False
    
    def load_image(self, image_path):
        """
        Load gambar menggunakan OpenCV
        
        Args:
            image_path (Path): Path ke file gambar
            
        Returns:
            numpy.ndarray: Image array atau None jika gagal
        """
        try:
            # Load dengan OpenCV
            image = cv2.imread(str(image_path))
            
            if image is None:
                self.logger.error(f"Gagal memuat gambar: {image_path}")
                return None
            
            # Convert BGR to RGB (OpenCV menggunakan BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.logger.info(f"Gambar dimuat: {image.shape[1]}x{image.shape[0]} pixels")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading gambar: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """
        Preprocess gambar untuk OCR yang lebih baik
        
        Args:
            image (numpy.ndarray): Image array
            
        Returns:
            numpy.ndarray: Processed image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Resize jika gambar terlalu kecil
            height, width = gray.shape
            if height < 300 or width < 300:
                scale = max(300/height, 300/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                self.logger.info(f"Gambar di-resize ke: {new_width}x{new_height}")
            
            # Noise reduction
            gray = cv2.medianBlur(gray, 3)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Sharpen image
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            self.logger.info("Preprocessing gambar selesai")
            return gray
            
        except Exception as e:
            self.logger.error(f"Error preprocessing gambar: {str(e)}")
            return image  # Return original jika preprocessing gagal
