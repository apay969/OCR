"""
Main file untuk menjalankan sistem EasyOCR
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import IMAGE_PATHS, LOG_FORMAT, LOG_LEVEL, LOGS_DIR
from src.ocr_processor import OCRProcessor
from utils.validation import validate_setup

def setup_logging():
    """Setup logging configuration"""
    log_file = LOGS_DIR / f"ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function"""
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("MEMULAI SISTEM EASYOCR")
    logger.info("=" * 50)
    
    try:
        # Validasi setup
        logger.info("Melakukan validasi setup...")
        if not validate_setup():
            logger.error("Validasi setup gagal!")
            return False
        
        # Inisialisasi OCR processor
        logger.info("Menginisialisasi OCR processor...")
        ocr = OCRProcessor()
        
        # Proses semua gambar
        logger.info(f"Memproses {len(IMAGE_PATHS)} gambar...")
        
        success_count = 0
        failed_count = 0
        
        for i, image_path in enumerate(IMAGE_PATHS, 1):
            logger.info(f"\n[{i}/{len(IMAGE_PATHS)}] Memproses: {image_path}")
            
            try:
                result = ocr.process_image(image_path)
                if result:
                    success_count += 1
                    logger.info(f"✅ Berhasil memproses: {Path(image_path).name}")
                else:
                    failed_count += 1
                    logger.warning(f"❌ Gagal memproses: {Path(image_path).name}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error memproses {Path(image_path).name}: {str(e)}")
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("RINGKASAN HASIL:")
        logger.info(f"✅ Berhasil: {success_count}")
        logger.info(f"❌ Gagal: {failed_count}")
        logger.info(f"📊 Total: {len(IMAGE_PATHS)}")
        logger.info("=" * 50)
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error dalam main function: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Memulai EasyOCR System...")
    print("📝 Cek log file untuk detail lengkap")
    print("-" * 50)
    
    success = main()
    
    if success:
        print("\n🎉 Proses selesai! Cek folder 'assets/output' untuk hasil.")
    else:
        print("\n❌ Proses gagal. Cek log untuk detail error.")
    
    input("\nTekan Enter untuk keluar...")
