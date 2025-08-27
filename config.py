"""
Konfigurasi aplikasi EasyOCR
"""
import os
from pathlib import Path

# Path konfigurasi
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
INPUT_DIR = ASSETS_DIR / "input"
OUTPUT_DIR = ASSETS_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Buat direktori jika belum ada
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# KONFIGURASI PATH GAMBAR
# ========================================
# GANTI PATH DI BAWAH INI SESUAI LOKASI GAMBAR ANDA!

# OPTION 1: Path Absolut (Ganti dengan path asli Anda)
IMAGE_PATHS = [
    # Windows - ganti dengan path Anda:
    r"D:\DATA DIRI\Magang\OCRNEW\ktp.png",
    # r"C:\Users\ADMIN\Documents\scan.png",
    # r"D:\Photos\screenshot.jpeg",
    
    # Linux/Mac - ganti dengan path Anda:
    # "/home/username/pictures/gambar1.jpg",
    # "/home/username/documents/scan.png",
    
    # OPTION 2: Gunakan folder assets/input (copy gambar ke sana)
    # str(INPUT_DIR / "contoh1.jpg"),   # Copy gambar ke assets/input/contoh1.jpg
    # str(INPUT_DIR / "contoh2.png"),   # Copy gambar ke assets/input/contoh2.png
    # str(INPUT_DIR / "contoh3.jpeg"),  # Copy gambar ke assets/input/contoh3.jpeg
    
    # TAMBAHKAN PATH GAMBAR ANDA DI SINI:
    # str(INPUT_DIR / "nama_gambar_anda.jpg"),
]

# ⚠️ PENTING: Hapus komentar (#) di depan path yang ingin digunakan!

# Konfigurasi OCR
OCR_CONFIG = {
    'languages': ['id', 'en'],  # Indonesian dan English
    'gpu': False,  # Set True jika ada GPU CUDA
    'detail': 1,   # 0=text only, 1=text+confidence
    'paragraph': False,
    'width_ths': 0.7,
    'height_ths': 0.7,
    'enhancement_level': 'aggressive',    # Untuk gambar buruk
    'confidence_threshold': 0.2,          # Terima confidence rendah  
    'enable_text_correction': True,
    'enable_word_dictionary': True
}

# Format output yang didukung
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
