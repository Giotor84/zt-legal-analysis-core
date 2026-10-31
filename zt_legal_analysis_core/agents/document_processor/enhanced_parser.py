# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Enhanced Document Parser - Multi-format extraction with LLM optimization
Combines PyMuPDF4LLM for PDFs with magic detection and robust fallbacks
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from threading import Thread
from queue import Queue, Empty

from zt_legal_analysis_core.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ParsedDocument:
    """Risultato parsing avanzato"""
    text: str
    metadata: Dict[str, Any]
    format: str
    tables: list = None
    images: list = None
    structure: Dict[str, Any] = None
    success: bool = True
    error: Optional[str] = None


class EnhancedDocumentParser:
    """
    Parser multi-formato ottimizzato per LLM e varietà documenti legali
    
    Strategie:
    - PDF: PyMuPDF4LLM (markdown strutturato) → PyMuPDF fallback
    - DOCX: python-docx con table detection
    - MSG/EML: extract-msg + libreria email standard
    - TXT/Generico: magic detection + encoding robusto
    """
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.doc': self._parse_doc,
            '.msg': self._parse_msg,
            '.eml': self._parse_eml,
            '.txt': self._parse_txt,
            '.xlsx': self._parse_xlsx,
            '.xls': self._parse_xls,
            '.csv': self._parse_csv
        }
        self._layout_available = None
        self._layout_module = None
        self._layout_error_logged = False
        
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse documento con rilevamento formato automatico
        
        Args:
            file_path: Percorso file da parsare
            
        Returns:
            ParsedDocument con text, metadata, structure
        """
        try:
            path = Path(file_path)
            
            # Verifica esistenza e dimensione
            if not path.exists():
                return ParsedDocument(
                    text="", 
                    metadata={}, 
                    format="unknown",
                    success=False,
                    error=f"File not found: {file_path}"
                )
            
            if path.stat().st_size < 10:
                return ParsedDocument(
                    text="", 
                    metadata={'size': path.stat().st_size}, 
                    format="empty",
                    success=False,
                    error="File too small (< 10 bytes)"
                )
            
            # Determina parser da estensione
            ext = path.suffix.lower()
            parser_func = self.supported_formats.get(ext, self._parse_generic)
            
            logger.info(f"Parsing {path.name} with format {ext}")
            return parser_func(file_path)
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format="error",
                success=False,
                error=str(e)
            )
    
    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """Parse PDF con PyMuPDF4LLM ottimizzato"""
        if not getattr(settings, "enable_pymupdf4llm", True):
            logger.info("PyMuPDF4LLM disabilitato, uso fallback PyMuPDF")
            return self._parse_pdf_fallback(file_path)

        try:
            import pymupdf4llm

            # PyMuPDF4LLM restituisce markdown strutturato ottimizzato per LLM
            md_text = self._call_with_timeout(
                pymupdf4llm.to_markdown,
                file_path,
                timeout=getattr(settings, "pdf_parse_timeout", 30.0)
            )

            if not md_text:
                raise ValueError("PyMuPDF4LLM returned empty text")
            
            # Estrai metadata base
            import fitz
            doc = fitz.open(file_path)
            metadata = {
                'pages': doc.page_count,
                'author': doc.metadata.get('author', ''),
                'title': doc.metadata.get('title', ''),
                'format': 'pdf'
            }
            doc.close()
            
            return ParsedDocument(
                text=md_text,
                metadata=metadata,
                format='pdf',
                structure={'markdown': True}
            )
            
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM failed, fallback to PyMuPDF: {str(e)}")
            return self._parse_pdf_fallback(file_path)

    def _call_with_timeout(self, func, *args, timeout: float = 30.0, **kwargs):
        """Esegue una funzione con timeout per evitare blocchi."""
        if not timeout or timeout <= 0:
            return func(*args, **kwargs)

        result_queue: Queue = Queue(maxsize=1)
        error_queue: Queue = Queue(maxsize=1)

        def _target():
            try:
                result_queue.put(func(*args, **kwargs))
            except Exception as e:
                error_queue.put(e)

        thread = Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Timeout parsing PDF dopo {timeout}s")

        try:
            err = error_queue.get_nowait()
            raise err
        except Empty:
            pass

        try:
            return result_queue.get_nowait()
        except Empty:
            return None
    
    def _parse_pdf_fallback(self, file_path: str) -> ParsedDocument:
        """Fallback con PyMuPDF standard"""
        try:
            import fitz

            doc = fitz.open(file_path)

            # Prova pdfplumber per layout migliorato
            if getattr(settings, "enable_pdfplumber_layout", True):
                plumber_text = self._extract_with_pdfplumber(file_path)
                if plumber_text:
                    metadata = {
                        'pages': doc.page_count,
                        'author': doc.metadata.get('author', ''),
                        'title': doc.metadata.get('title', ''),
                        'format': 'pdf',
                        'layout': 'pdfplumber'
                    }
                    doc.close()
                    return ParsedDocument(
                        text=plumber_text,
                        metadata=metadata,
                        format='pdf',
                        structure={'layout': True}
                    )

            # Prova pymupdf_layout se abilitato
            if getattr(settings, "enable_pymupdf_layout", True):
                layout_text = self._extract_with_layout(file_path)
                if layout_text:
                    metadata = {
                        'pages': doc.page_count,
                        'author': doc.metadata.get('author', ''),
                        'title': doc.metadata.get('title', ''),
                        'format': 'pdf',
                        'layout': 'pymupdf_layout'
                    }
                    doc.close()
                    return ParsedDocument(
                        text=layout_text,
                        metadata=metadata,
                        format='pdf',
                        structure={'layout': True}
                    )

            text_parts = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_parts.append(page.get_text())

            metadata = {
                'pages': doc.page_count,
                'author': doc.metadata.get('author', ''),
                'title': doc.metadata.get('title', ''),
                'format': 'pdf'
            }

            doc.close()

            return ParsedDocument(
                text="\n\n".join(text_parts),
                metadata=metadata,
                format='pdf'
            )
            
        except Exception as e:
            logger.error(f"PDF fallback failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='pdf',
                success=False,
                error=str(e)
            )

    def _extract_with_pdfplumber(self, file_path: str) -> Optional[str]:
        """Estrae testo usando pdfplumber se disponibile."""
        try:
            import pdfplumber

            texts = []
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            texts.append(f"--- PAGINA {idx + 1} ---\n{page_text}")
                    except Exception:
                        continue

            return "\n\n".join(texts) if texts else None
        except Exception as e:
            if not self._layout_error_logged:
                logger.warning(f"pdfplumber non disponibile o fallito: {e}")
                self._layout_error_logged = True
            return None

    def _extract_with_layout(self, file_path: str) -> Optional[str]:
        """Estrae testo usando pymupdf_layout se disponibile."""
        if self._layout_available is False:
            return None

        try:
            if self._layout_module is None:
                import importlib

                last_error = None
                for module_name in (
                    "pymupdf_layout",
                    "pymupdf_layout.layout",
                    "pymupdf_layout.main",
                ):
                    try:
                        self._layout_module = importlib.import_module(module_name)
                        self._layout_available = True
                        break
                    except Exception as exc:
                        last_error = exc

                if self._layout_module is None:
                    raise last_error or ImportError("pymupdf_layout non disponibile")

            # Supporta diversi nomi funzione per compatibilità
            for attr in (
                "extract_text_from_file",
                "extract_text",
                "to_text",
                "get_text",
                "extract_text_from_pdf",
            ):
                func = getattr(self._layout_module, attr, None)
                if callable(func):
                    try:
                        result = func(file_path)
                        if isinstance(result, str) and result.strip():
                            return result
                    except Exception:
                        pass

            # Fallback: cerca funzioni con nome compatibile
            import inspect

            for name, func in inspect.getmembers(self._layout_module, inspect.isfunction):
                lname = name.lower()
                if "text" not in lname and "extract" not in lname:
                    continue
                try:
                    sig = inspect.signature(func)
                    if len(sig.parameters) >= 1:
                        result = func(file_path)
                        if isinstance(result, str) and result.strip():
                            return result
                except Exception:
                    continue

            if not self._layout_error_logged:
                logger.warning("pymupdf_layout disponibile ma nessuna funzione di estrazione compatibile trovata")
                self._layout_error_logged = True
            return None
        except Exception as e:
            self._layout_available = False
            if not self._layout_error_logged:
                logger.debug(f"pymupdf_layout non disponibile o fallito: {e}")
                self._layout_error_logged = True
            return None
    
    def _parse_docx(self, file_path: str) -> ParsedDocument:
        """Parse DOCX con table detection"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            # Estrai paragrafi
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Estrai tabelle
            tables_data = []
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    table_text.append(" | ".join(row_text))
                tables_data.append("\n".join(table_text))
            
            # Combina testo
            full_text = "\n\n".join(paragraphs)
            if tables_data:
                full_text += "\n\n=== TABELLE ===\n\n" + "\n\n".join(tables_data)
            
            metadata = {
                'paragraphs': len(paragraphs),
                'tables': len(tables_data),
                'format': 'docx'
            }
            
            return ParsedDocument(
                text=full_text,
                metadata=metadata,
                format='docx',
                tables=tables_data
            )
            
        except Exception as e:
            logger.error(f"DOCX parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='docx',
                success=False,
                error=str(e)
            )
    
    def _parse_doc(self, file_path: str) -> ParsedDocument:
        """Parse DOC (legacy) - requires conversion"""
        # Legacy DOC richiede LibreOffice/antiword
        # Per ora, ritorna errore informativo
        return ParsedDocument(
            text="",
            metadata={},
            format='doc',
            success=False,
            error="Legacy .doc format not supported. Please convert to .docx"
        )
    
    def _parse_msg(self, file_path: str) -> ParsedDocument:
        """Parse MSG con extract-msg"""
        try:
            import extract_msg
            
            msg = extract_msg.Message(file_path)
            
            # Estrai campi principali
            text_parts = []
            text_parts.append(f"Da: {msg.sender}")
            text_parts.append(f"A: {msg.to}")
            text_parts.append(f"Oggetto: {msg.subject}")
            text_parts.append(f"Data: {msg.date}")
            text_parts.append("\n" + msg.body)
            
            # Estrai allegati
            attachments = [att.longFilename for att in msg.attachments]
            
            metadata = {
                'sender': msg.sender,
                'to': msg.to,
                'subject': msg.subject,
                'date': str(msg.date),
                'attachments': attachments,
                'format': 'msg'
            }
            
            return ParsedDocument(
                text="\n".join(text_parts),
                metadata=metadata,
                format='msg'
            )
            
        except Exception as e:
            logger.error(f"MSG parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='msg',
                success=False,
                error=str(e)
            )
    
    def _parse_eml(self, file_path: str) -> ParsedDocument:
        """Parse EML con libreria email standard (built-in Python)"""
        try:
            import email
            from email import policy
            
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
            
            # Estrai header
            text_parts = []
            text_parts.append(f"Da: {msg.get('From', 'N/A')}")
            text_parts.append(f"A: {msg.get('To', 'N/A')}")
            text_parts.append(f"CC: {msg.get('Cc', 'N/A')}")
            text_parts.append(f"Oggetto: {msg.get('Subject', 'N/A')}")
            text_parts.append(f"Data: {msg.get('Date', 'N/A')}")
            
            # Estrai corpo
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body = part.get_content()
                        break
                    elif content_type == "text/html" and not body:
                        # Fallback a HTML se non c'è plain text
                        html_body = part.get_content()
                        # Rimuovi tag HTML basilari
                        import re
                        body = re.sub(r'<[^>]+>', ' ', html_body)
            else:
                body = msg.get_content()
            
            text_parts.append("\n" + body)
            
            # Lista allegati
            attachments = []
            if msg.is_multipart():
                for part in msg.walk():
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)
            
            metadata = {
                'sender': msg.get('From', ''),
                'to': msg.get('To', ''),
                'subject': msg.get('Subject', ''),
                'date': msg.get('Date', ''),
                'attachments': attachments,
                'format': 'eml'
            }
            
            return ParsedDocument(
                text="\n".join(text_parts),
                metadata=metadata,
                format='eml'
            )
            
        except Exception as e:
            logger.error(f"EML parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='eml',
                success=False,
                error=str(e)
            )
    
    def _parse_txt(self, file_path: str) -> ParsedDocument:
        """Parse TXT con encoding detection"""
        try:
            # Prova encoding comuni
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            text = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    used_encoding = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if text is None:
                # Fallback binario
                with open(file_path, 'rb') as f:
                    text = f.read().decode('utf-8', errors='ignore')
                used_encoding = 'utf-8-ignore'
            
            metadata = {
                'encoding': used_encoding,
                'size': len(text),
                'format': 'txt'
            }
            
            return ParsedDocument(
                text=text,
                metadata=metadata,
                format='txt'
            )
            
        except Exception as e:
            logger.error(f"TXT parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='txt',
                success=False,
                error=str(e)
            )
    
    def _parse_xlsx(self, file_path: str) -> ParsedDocument:
        """Parse XLSX con openpyxl"""
        try:
            from openpyxl import load_workbook
            
            wb = load_workbook(file_path, read_only=True)
            
            sheets_text = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheet_text = [f"=== {sheet_name} ==="]
                
                for row in ws.iter_rows(values_only=True):
                    row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                    if row_text.strip():
                        sheet_text.append(row_text)
                
                sheets_text.append("\n".join(sheet_text))
            
            full_text = "\n\n".join(sheets_text)
            
            metadata = {
                'sheets': len(wb.sheetnames),
                'sheet_names': wb.sheetnames,
                'format': 'xlsx'
            }
            
            return ParsedDocument(
                text=full_text,
                metadata=metadata,
                format='xlsx'
            )
            
        except Exception as e:
            logger.error(f"XLSX parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='xlsx',
                success=False,
                error=str(e)
            )
    
    def _parse_xls(self, file_path: str) -> ParsedDocument:
        """Parse XLS (legacy) - richiede xlrd"""
        return ParsedDocument(
            text="",
            metadata={},
            format='xls',
            success=False,
            error="Legacy .xls format not supported. Please convert to .xlsx"
        )
    
    def _parse_csv(self, file_path: str) -> ParsedDocument:
        """Parse CSV con pandas"""
        try:
            import pandas as pd
            
            # Prova delimiter comuni
            for delimiter in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            
            # Converti in testo tabellare
            text = df.to_string(index=False)
            
            metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'format': 'csv'
            }
            
            return ParsedDocument(
                text=text,
                metadata=metadata,
                format='csv'
            )
            
        except Exception as e:
            logger.error(f"CSV parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='csv',
                success=False,
                error=str(e)
            )
    
    def _parse_generic(self, file_path: str) -> ParsedDocument:
        """Fallback generico con magic detection"""
        try:
            import magic
            
            # Rileva MIME type
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            
            # Se è testo, prova parsing
            if file_type.startswith('text/'):
                return self._parse_txt(file_path)
            
            # Altrimenti errore
            return ParsedDocument(
                text="",
                metadata={'mime_type': file_type},
                format='unknown',
                success=False,
                error=f"Unsupported file type: {file_type}"
            )
            
        except Exception as e:
            logger.error(f"Generic parsing failed: {str(e)}")
            return ParsedDocument(
                text="",
                metadata={},
                format='unknown',
                success=False,
                error=str(e)
            )


# Singleton instance
_parser_instance = None

def get_parser() -> EnhancedDocumentParser:
    """Get singleton parser instance"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = EnhancedDocumentParser()
    return _parser_instance
