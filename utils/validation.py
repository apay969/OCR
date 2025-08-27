"""
Validation utility - Validasi setup dan requirements
"""
import logging
import sys
from pathlib import Path

from config import IMAGE_PATHS, INPUT_DIR, OUTPUT_DIR, LOGS_DIR

def validate_setup():
    """
    Validasi setup aplikasi
    
    Returns:
        bool: True jika setup valid, False jika tidak
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validasi Python version
        if sys.version_info < (3, 7):
            logger.error(f"Python version terlalu lama: {sys.version}")
            logger.error("Minimal Python 3.7 diperlukan")
            return False
        
        # Validasi dependencies
        if not validate_dependencies():
            return False
        
        # Validasi direktori
        if not validate_directories():
            return False
        
        # Validasi image paths
        if not validate_image_paths():
            return False
        
        logger.info("✅ Semua validasi berhasil!")
        return True
        
    except Exception as e:
        logger.error(f"Error dalam validasi setup: {str(e)}")
        return False

def validate_dependencies():
    """Validasi dependencies yang diperlukan"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'easyocr',
        'cv2',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'easyocr':
                import easyocr
            elif package == 'numpy':
                import numpy
            
            logger.info(f"✅ {package} tersedia")
            
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} tidak ditemukan")
    
    if missing_packages:
        logger.error("Install dependencies yang hilang dengan:")
        logger.error("pip install -r requirements.txt")
        return False
    
    return True

def validate_directories():
    """Validasi direktori yang diperlukan"""
    logger = logging.getLogger(__name__)
    
    directories = [
        ("Input", INPUT_DIR),
        ("Output", OUTPUT_DIR),
        ("Logs", LOGS_DIR)
    ]
    
    for name, directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.exists() and directory.is_dir():
                logger.info(f"✅ Direktori {name}: {directory}")
            else:
                logger.error(f"❌ Gagal membuat direktori {name}: {directory}")
                return False
        except Exception as e:
            logger.error(f"❌ Error direktori {name}: {str(e)}")
            return False
    
    return True

def validate_image_paths():
    """Validasi path gambar yang dikonfigurasi"""
    logger = logging.getLogger(__name__)
    
    if not IMAGE_PATHS:
        logger.error("❌ Tidak ada path gambar yang dikonfigurasi!")
        logger.error("Edit config.py dan tambahkan path gambar di IMAGE_PATHS")
        return False
    
    valid_paths = 0
    
    for i, image_path in enumerate(IMAGE_PATHS, 1):
        path = Path(image_path)
        
        if path.exists():
            logger.info(f"✅ [{i}] {path.name} - ditemukan")
            valid_paths += 1
        else:
            logger.warning(f"⚠️  [{i}] {path.name} - tidak ditemukan di: {path}")
    
    if valid_paths == 0:
        logger.error("❌ Tidak ada gambar yang ditemukan!")
        logger.error("Pastikan path gambar di config.py sudah benar")
        logger.error("Atau copy gambar ke folder: assets/input/")
        return False
    
    logger.info(f"📊 {valid_paths}/{len(IMAGE_PATHS)} gambar ditemukan")
    return True
