"""
Enhanced Image Handler - Menangani loading dan preprocessing gambar untuk OCR yang lebih akurat
Versi sederhana yang fokus pada perbaikan masalah OCR dengan penyempurnaan kecil
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
        Validasi file gambar dengan validasi tambahan
        
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
            
            # Validasi tambahan: coba buka file untuk memastikan tidak corrupt
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                self.logger.error(f"File gambar corrupt atau tidak valid: {image_path}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validasi gambar: {str(e)}")
            return False
    
    def load_image(self, image_path):
        """
        Load gambar dengan fallback ke PIL jika OpenCV gagal
        
        Args:
            image_path (Path): Path ke file gambar
            
        Returns:
            numpy.ndarray: Image array atau None jika gagal
        """
        try:
            # Coba load dengan OpenCV dulu
            image = cv2.imread(str(image_path))
            
            if image is None:
                # Fallback ke PIL jika OpenCV gagal
                self.logger.warning(f"OpenCV gagal, mencoba PIL: {image_path}")
                try:
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    # Jika PIL berhasil, konversi ke format OpenCV
                    if len(image.shape) == 3:
                        if image.shape[2] == 4:  # RGBA
                            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                        elif image.shape[2] == 3:  # RGB
                            pass  # sudah RGB
                except Exception as pil_error:
                    self.logger.error(f"PIL juga gagal: {str(pil_error)}")
                    return None
            else:
                # Convert BGR to RGB (OpenCV menggunakan BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is not None:
                self.logger.info(f"Gambar dimuat: {image.shape[1]}x{image.shape[0]} pixels")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading gambar: {str(e)}")
            return None
    
    def detect_orientation(self, image):
        """
        Deteksi orientasi gambar yang lebih sederhana dan reliable
        
        Args:
            image (numpy.ndarray): Image array
            
        Returns:
            numpy.ndarray: Rotated image
        """
        try:
            # Convert to grayscale jika belum
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Hough lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None and len(lines) > 3:
                angles = []
                for rho, theta in lines[:10]:
                    angle = np.degrees(theta)
                    # Normalize angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                if angles:
                    # Ambil median angle
                    median_angle = np.median(angles)
                    # Hanya rotasi jika angle signifikan
                    if abs(median_angle) > 2:
                        self.logger.info(f"Merotasi gambar sebesar {median_angle:.1f} derajat")
                        return self.rotate_image(image, -median_angle)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error deteksi orientasi: {str(e)}")
            return image
    
    def rotate_image(self, image, angle):
        """
        Rotasi gambar sederhana
        
        Args:
            image (numpy.ndarray): Image array
            angle (float): Sudut rotasi dalam derajat
            
        Returns:
            numpy.ndarray: Rotated image
        """
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Matriks rotasi
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotasi gambar
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
            
            return rotated
            
        except Exception as e:
            self.logger.error(f"Error rotasi gambar: {str(e)}")
            return image
    
    def enhance_contrast_adaptive(self, image):
        """
        Adaptive contrast enhancement yang lebih baik untuk teks
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        try:
            # CLAHE dengan parameter optimal untuk teks
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
            
            # Gamma correction yang lebih halus
            gamma = 1.1
            gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
            gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
            
            return gamma_corrected
            
        except Exception as e:
            self.logger.error(f"Error enhance contrast: {str(e)}")
            return image
    
    def reduce_noise_advanced(self, image):
        """
        Noise reduction yang lebih canggih untuk preservasi karakter
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        try:
            # Bilateral filter dengan parameter yang lebih halus
            denoised = cv2.bilateralFilter(image, 7, 50, 50)
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"Error reduce noise: {str(e)}")
            return image
    
    def sharpen_text(self, image):
        """
        Text sharpening yang lebih targeted untuk karakter
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Sharpened image
        """
        try:
            # Unsharp mask untuk meningkatkan ketajaman teks
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            
            # Clamp values
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"Error sharpen text: {str(e)}")
            return image
    
    def preprocess_image(self, image, enhance_text=True, auto_rotate=True):
        """
        Preprocessing yang disempurnakan untuk OCR yang lebih akurat
        
        Args:
            image (numpy.ndarray): Image array
            enhance_text (bool): Apakah menggunakan text enhancement
            auto_rotate (bool): Apakah menggunakan auto rotation
            
        Returns:
            numpy.ndarray: Processed image
        """
        try:
            original_image = image.copy()
            
            # Auto rotation jika diminta
            if auto_rotate:
                image = self.detect_orientation(image)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Resize jika gambar terlalu kecil - optimal size
            height, width = gray.shape
            min_size = 900  # Sweet spot untuk OCR
            
            if height < min_size or width < min_size:
                scale = max(min_size/height, min_size/width)
                # Batasi scale untuk hasil optimal
                scale = min(scale, 2.0)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                self.logger.info(f"Gambar di-resize ke: {new_width}x{new_height}")
            
            if enhance_text:
                # Advanced noise reduction
                gray = self.reduce_noise_advanced(gray)
                
                # Adaptive contrast enhancement
                gray = self.enhance_contrast_adaptive(gray)
                
                # Text sharpening
                gray = self.sharpen_text(gray)
            else:
                # Minimal noise reduction jika tidak enhance
                gray = cv2.medianBlur(gray, 3)
            
            # Final clamp
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            
            self.logger.info("Enhanced preprocessing selesai")
            return gray
            
        except Exception as e:
            self.logger.error(f"Error preprocessing gambar: {str(e)}")
            return original_image if 'original_image' in locals() else image
    
    def preprocess_for_ocr_clean(self, image):
        """
        Preprocessing khusus untuk mendapatkan teks yang bersih
        Fokus pada binary threshold untuk OCR yang lebih akurat dengan penyempurnaan
        
        Args:
            image (numpy.ndarray): Image array
            
        Returns:
            numpy.ndarray: Clean binary image
        """
        try:
            # Convert to grayscale dulu
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Resize untuk kualitas OCR yang optimal
            height, width = gray.shape
            if height < 900 or width < 900:
                scale = max(900/height, 900/width)
                scale = min(scale, 1.8)  # Moderate scaling
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Noise reduction yang balanced
            gray = cv2.bilateralFilter(gray, 9, 60, 60)
            
            # Adaptive threshold dengan parameter yang fine-tuned
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 9, 3)
            
            # Morphological cleaning yang minimal tapi efektif
            kernel = np.ones((1,1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            self.logger.info("Clean OCR preprocessing selesai")
            return binary
            
        except Exception as e:
            self.logger.error(f"Error clean preprocessing: {str(e)}")
            return image
    
    def get_image_quality_score(self, image):
        """
        Enhanced quality assessment
        
        Args:
            image (numpy.ndarray): Image array
            
        Returns:
            dict: Quality metrics
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Blur detection - lebih sensitif
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast - standard deviation
            contrast_score = gray.std()
            
            # Brightness
            brightness_score = gray.mean()
            
            # Resolution
            resolution_score = gray.shape[0] * gray.shape[1]
            
            # Text clarity check - gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            text_clarity = edge_magnitude.mean()
            
            # Improved quality assessment
            is_low_quality = (blur_score < 150 or 
                             contrast_score < 40 or 
                             text_clarity < 20 or
                             resolution_score < 400000)  # < 800x500
            
            metrics = {
                'blur_score': blur_score,
                'contrast_score': contrast_score,
                'brightness_score': brightness_score,
                'resolution_score': resolution_score,
                'text_clarity': text_clarity,
                'is_low_quality': is_low_quality
            }
            
            self.logger.info(f"Quality metrics: blur={blur_score:.1f}, contrast={contrast_score:.1f}, clarity={text_clarity:.1f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return {'is_low_quality': False}
