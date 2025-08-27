"""
Text Processor - Memproses hasil OCR text
"""
import logging
import re
from config import OCR_CONFIG

class TextProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_results(self, ocr_results):
        """
        Memproses hasil OCR menjadi text yang bersih
        
        Args:
            ocr_results (list): Hasil dari EasyOCR
            
        Returns:
            str: Text yang sudah diproses
        """
        try:
            if not ocr_results:
                return ""
            
            # Extract text berdasarkan format hasil OCR
            texts = []
            
            for result in ocr_results:
                if OCR_CONFIG['detail'] == 1:
                    # Format: [bbox, text, confidence]
                    bbox, text, confidence = result
                    
                    # Filter berdasarkan confidence threshold
                    if confidence >= 0.5:  # Minimal 50% confidence
                        texts.append(text.strip())
                    else:
                        self.logger.debug(f"Text diabaikan (confidence rendah): '{text}' ({confidence:.2f})")
                else:
                    # Format: text only
                    texts.append(result.strip())
            
            # Gabungkan semua text
            raw_text = ' '.join(texts)
            
            # Clean dan normalize text
            cleaned_text = self._clean_text(raw_text)
            
            self.logger.info(f"Text processing selesai: {len(texts)} segmen text")
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return ""
    
    def _clean_text(self, text):
        """
        Membersihkan dan normalize text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        try:
            if not text:
                return ""
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text)
            
            # Fix common OCR errors
            text = self._fix_common_ocr_errors(text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def _fix_common_ocr_errors(self, text):
        """
        Perbaiki kesalahan OCR yang umum terjadi
        
        Args:
            text (str): Input text
            
        Returns:
            str: Corrected text
        """
        try:
            # Common OCR character mistakes
            corrections = {
                # Numbers and letters
                '0': ['O', 'o'],
                '1': ['l', 'I', '|'],
                '5': ['S'],
                '6': ['G'],
                '8': ['B'],
                
                # Letters
                'O': ['0'],
                'I': ['1', '|'],
                'l': ['1', 'I'],
                'S': ['5'],
                'G': ['6'],
                'B': ['8'],
                
                # Special characters
                '"': ['"', '"'],
                "'": [''', '''],
            }
            
            # Apply corrections contextually
            corrected_text = text
            
            # Fix spacing around punctuation
            corrected_text = re.sub(r'\s+([.,:;!?])', r'\1', corrected_text)
            corrected_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', corrected_text)
            
            return corrected_text
            
        except Exception as e:
            self.logger.error(f"Error fixing OCR errors: {str(e)}")
            return text
