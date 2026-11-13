# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
GPU Performance Monitor

Monitora utilizzo GPU durante elaborazione NPL.
"""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class GPUStats:
    """Statistiche GPU"""
    utilization: int  # %
    memory_used: int  # MB
    memory_total: int  # MB
    temperature: Optional[int] = None  # °C
    
    @property
    def memory_utilization(self) -> float:
        """Percentuale memoria usata"""
        return (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0
    
    def __str__(self) -> str:
        return f"GPU: {self.utilization}% | VRAM: {self.memory_used}/{self.memory_total}MB ({self.memory_utilization:.0f}%)"


class GPUMonitor:
    """Monitora GPU NVIDIA"""
    
    def __init__(self):
        self.available = self._check_nvidia_smi()
    
    def _check_nvidia_smi(self) -> bool:
        """Verifica se nvidia-smi è disponibile"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True, timeout=2)
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Optional[GPUStats]:
        """Ottiene statistiche GPU correnti"""
        if not self.available:
            return None
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return GPUStats(
                    utilization=int(parts[0].strip()),
                    memory_used=int(parts[1].strip()),
                    memory_total=int(parts[2].strip()),
                    temperature=int(parts[3].strip()) if len(parts) > 3 else None
                )
        except Exception as e:
            logger.debug(f"GPU stats failed: {e}")
        
        return None
    
    def log_stats(self):
        """Log statistiche GPU"""
        stats = self.get_stats()
        if stats:
            temp_str = f" | {stats.temperature}°C" if stats.temperature else ""
            logger.info(f"   📊 {stats}{temp_str}")
    
    def is_gpu_saturated(self, threshold: float = 95.0) -> bool:
        """Verifica se GPU è saturata"""
        stats = self.get_stats()
        return stats.utilization >= threshold if stats else False

    def cleanup(self):
        """Pulizia VRAM best-effort (no-op se torch non disponibile)."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("   🧹 VRAM cache pulita")
        except Exception as e:
            logger.debug(f"GPU cleanup skipped: {e}")


# Singleton globale
_gpu_monitor = None

def get_gpu_monitor() -> GPUMonitor:
    """Ottiene GPU monitor singleton"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor
