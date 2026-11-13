# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Image Preprocessor - Preprocessing OpenCV per documenti scansionati

Ottimizzazioni per OCR:
1. Binarizzazione adattiva (Otsu/Sauvola)
2. Deskew (correzione rotazione)
3. Rimozione bordi neri
4. Denoising
5. Contrast enhancement
"""

from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import numpy as np


@dataclass
class PreprocessingResult:
    """Risultato preprocessing immagine"""
    image: np.ndarray
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    rotation_angle: float = 0.0
    binarized: bool = False
    denoised: bool = False
    quality_improvement: float = 0.0  # Stima miglioramento qualità


class ImagePreprocessor:
    """
    Preprocessore immagini per migliorare risultati OCR.
    
    Ottimizzato per documenti bancari scansionati:
    - Estratti conto con tabelle
    - Contratti con testo fitto
    - Documenti con timbri e firme
    """
    
    def __init__(self):
        self._cv2 = None
        self._ensure_opencv()
    
    def _ensure_opencv(self) -> bool:
        """Carica OpenCV se disponibile"""
        if self._cv2 is not None:
            return True
        
        try:
            import cv2
            self._cv2 = cv2
            logger.info("✅ OpenCV disponibile per preprocessing")
            return True
        except ImportError:
            logger.warning("⚠️ OpenCV non disponibile - pip install opencv-python")
            return False
    
    def preprocess(
        self,
        image: Union[np.ndarray, "PIL.Image.Image"],
        auto_deskew: bool = True,
        binarize: bool = True,
        denoise: bool = True,
        remove_borders: bool = True,
        enhance_contrast: bool = True
    ) -> PreprocessingResult:
        """
        Applica preprocessing completo all'immagine.
        
        Args:
            image: Immagine (numpy array o PIL Image)
            auto_deskew: Correggi rotazione automaticamente
            binarize: Applica binarizzazione adattiva
            denoise: Rimuovi rumore
            remove_borders: Rimuovi bordi neri da scanner
            enhance_contrast: Migliora contrasto
            
        Returns:
            PreprocessingResult con immagine preprocessata
        """
        # Converti a numpy array se PIL
        if hasattr(image, 'mode'):  # PIL Image
            img = np.array(image)
        else:
            img = image.copy()
        
        original_shape = img.shape[:2]
        
        # Fallback se OpenCV non disponibile
        if not self._ensure_opencv():
            return PreprocessingResult(
                image=img,
                original_shape=original_shape,
                processed_shape=original_shape
            )
        
        cv2 = self._cv2
        rotation_angle = 0.0
        binarized = False
        denoised = False
        
        try:
            # 1. Converti a grayscale se necessario
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # 2. Rimuovi bordi neri
            if remove_borders:
                gray = self._remove_black_borders(gray)
            
            # 3. Deskew (correzione rotazione)
            if auto_deskew:
                gray, rotation_angle = self._deskew(gray)
            
            # 4. Enhance contrast (CLAHE)
            if enhance_contrast:
                gray = self._enhance_contrast(gray)
            
            # 5. Denoise
            if denoise:
                gray = self._denoise(gray)
                denoised = True
            
            # 6. Binarizzazione adattiva
            if binarize:
                gray = self._adaptive_binarize(gray)
                binarized = True
            
            # Stima miglioramento qualità
            quality_improvement = self._estimate_improvement(
                np.array(image) if hasattr(image, 'mode') else image,
                gray
            )
            
            return PreprocessingResult(
                image=gray,
                original_shape=original_shape,
                processed_shape=gray.shape[:2],
                rotation_angle=rotation_angle,
                binarized=binarized,
                denoised=denoised,
                quality_improvement=quality_improvement
            )
            
        except Exception as e:
            logger.error(f"Errore preprocessing: {e}")
            return PreprocessingResult(
                image=img,
                original_shape=original_shape,
                processed_shape=original_shape
            )
    
    def _remove_black_borders(self, img: np.ndarray, threshold: int = 20) -> np.ndarray:
        """Rimuove bordi neri tipici degli scanner"""
        cv2 = self._cv2
        
        try:
            # Trova contorni non neri
            _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            
            # Trova bounding box del contenuto
            coords = cv2.findNonZero(binary)
            if coords is None:
                return img
            
            x, y, w, h = cv2.boundingRect(coords)
            
            # Aggiungi margine
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            cropped = img[y:y+h, x:x+w]
            
            # Non restituire immagine troppo piccola
            if cropped.shape[0] < 100 or cropped.shape[1] < 100:
                return img
            
            return cropped
            
        except Exception as e:
            logger.debug(f"Border removal fallito: {e}")
            return img
    
    def _deskew(self, img: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
        """
        Corregge la rotazione del documento.
        
        Usa trasformata di Hough per rilevare linee e calcolare angolo.
        """
        cv2 = self._cv2
        
        try:
            # Edge detection
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            
            # Hough lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=img.shape[1] // 4,
                maxLineGap=20
            )
            
            if lines is None or len(lines) == 0:
                return img, 0.0
            
            # Calcola angolo medio delle linee orizzontali
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):  # Linea principalmente orizzontale
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < max_angle:
                        angles.append(angle)
            
            if not angles:
                return img, 0.0
            
            # Mediana degli angoli (più robusta alla mean)
            skew_angle = np.median(angles)
            
            # Se angolo troppo piccolo, non ruotare
            if abs(skew_angle) < 0.5:
                return img, 0.0
            
            # Ruota immagine
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Calcola nuove dimensioni per evitare crop
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            
            # Aggiusta matrice per centrare
            rotation_matrix[0, 2] += (new_w - w) / 2
            rotation_matrix[1, 2] += (new_h - h) / 2
            
            rotated = cv2.warpAffine(
                img,
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.debug(f"Deskew: rotazione {skew_angle:.2f}°")
            return rotated, skew_angle
            
        except Exception as e:
            logger.debug(f"Deskew fallito: {e}")
            return img, 0.0
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Migliora contrasto con CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        cv2 = self._cv2
        
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
            return enhanced
        except Exception as e:
            logger.debug(f"Contrast enhancement fallito: {e}")
            return img
    
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Rimuove rumore preservando bordi (fastNlMeansDenoising)"""
        cv2 = self._cv2
        
        try:
            # Fast Non-Local Means Denoising
            denoised = cv2.fastNlMeansDenoising(
                img,
                h=10,          # Filtro strength
                templateWindowSize=7,
                searchWindowSize=21
            )
            return denoised
        except Exception as e:
            logger.debug(f"Denoising fallito: {e}")
            return img
    
    def _adaptive_binarize(
        self,
        img: np.ndarray,
        method: str = "sauvola"
    ) -> np.ndarray:
        """
        Binarizzazione adattiva.
        
        Metodi:
        - "otsu": Soglia globale automatica
        - "adaptive": Soglia locale (buono per illuminazione non uniforme)
        - "sauvola": Migliore per documenti (raccomandato)
        """
        cv2 = self._cv2
        
        try:
            if method == "otsu":
                _, binary = cv2.threshold(
                    img, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                return binary
            
            elif method == "adaptive":
                binary = cv2.adaptiveThreshold(
                    img, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=31,
                    C=10
                )
                return binary
            
            elif method == "sauvola":
                # Sauvola: migliore per documenti
                # Formula: T(x,y) = mean * (1 + k * (std/R - 1))
                try:
                    from skimage.filters import threshold_sauvola
                    threshold = threshold_sauvola(img, window_size=31, k=0.2)
                    binary = (img > threshold).astype(np.uint8) * 255
                    return binary
                except ImportError:
                    # Fallback a adaptive se skimage non disponibile
                    logger.debug("skimage non disponibile, uso adaptive threshold")
                    return self._adaptive_binarize(img, method="adaptive")
            
            else:
                return img
                
        except Exception as e:
            logger.debug(f"Binarizzazione fallita: {e}")
            return img
    
    def _estimate_improvement(
        self,
        original: np.ndarray,
        processed: np.ndarray
    ) -> float:
        """
        Stima miglioramento qualità dopo preprocessing.
        
        Basato su:
        - Aumento contrasto (deviazione standard)
        - Riduzione rumore
        """
        try:
            # Converti original a grayscale se necessario
            if len(original.shape) == 3:
                orig_gray = np.mean(original, axis=2)
            else:
                orig_gray = original
            
            # Confronta deviazione standard (contrasto)
            orig_std = np.std(orig_gray)
            proc_std = np.std(processed)
            
            if orig_std > 0:
                contrast_improvement = (proc_std - orig_std) / orig_std
            else:
                contrast_improvement = 0
            
            # Limita a range ragionevole
            improvement = max(-0.5, min(1.0, contrast_improvement))
            
            return round(improvement, 2)
            
        except Exception:
            return 0.0
    
    def preprocess_for_ocr(
        self,
        image: Union[np.ndarray, "PIL.Image.Image"],
        is_table: bool = False
    ) -> np.ndarray:
        """
        Preprocessing ottimizzato per OCR.
        
        Args:
            image: Immagine da preprocessare
            is_table: True se documento tabulare (usa binarizzazione meno aggressiva)
            
        Returns:
            Immagine preprocessata come numpy array
        """
        result = self.preprocess(
            image,
            auto_deskew=True,
            binarize=not is_table,  # Per tabelle, evita binarizzazione forte
            denoise=True,
            remove_borders=True,
            enhance_contrast=True
        )
        
        return result.image


# Singleton
_preprocessor: Optional[ImagePreprocessor] = None


def get_preprocessor() -> ImagePreprocessor:
    """Ottiene istanza singleton del preprocessor"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor()
    return _preprocessor
