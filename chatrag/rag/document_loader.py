import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import tempfile
import uuid

class DocumentLoader:
    """
    Utility class for loading and processing documents from various file formats.
    """
    
    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize the document loader.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
    async def save_uploaded_file(self, file) -> str:
        """
        Save an uploaded file to disk and return its path.
        
        Args:
            file: The uploaded file object from FastAPI
            
        Returns:
            Path to the saved file
        """
        # Create a unique filename to avoid collisions
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = self.upload_dir / filename
        
        # Write the file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return str(file_path)
    
    async def load_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load a document and split it into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == ".txt":
                return await self._load_text(file_path)
            elif file_ext == ".pdf":
                return await self._load_pdf(file_path)
            elif file_ext in [".docx", ".doc"]:
                return await self._load_docx(file_path)
            elif file_ext in [".csv", ".xlsx", ".xls"]:
                return await self._load_tabular(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return []
            
    async def _load_text(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a text file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Simple chunking by paragraphs
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        return [
            {
                "content": chunk,
                "metadata": {
                    "source": Path(file_path).name,
                    "chunk_id": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
    async def _load_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a PDF file.
        """
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            text_chunks = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_chunks.append({
                        "content": text.strip(),
                        "metadata": {
                            "source": Path(file_path).name,
                            "page": i + 1,
                            "chunk_id": i
                        }
                    })
                    
            return text_chunks
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: pip install pypdf")
            
    async def _load_docx(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a Word document.
        """
        try:
            import docx
            
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            return [
                {
                    "content": para,
                    "metadata": {
                        "source": Path(file_path).name,
                        "chunk_id": i
                    }
                }
                for i, para in enumerate(paragraphs)
            ]
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
            
    async def _load_tabular(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and process tabular data (CSV, Excel).
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            import pandas as pd
            
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
                
            # Convert each row to a text chunk
            chunks = []
            for i, row in enumerate(df.to_dict("records")):
                # Format the row as a string
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                chunks.append({
                    "content": content,
                    "metadata": {
                        "source": Path(file_path).name,
                        "row": i + 1,
                        "chunk_id": i
                    }
                })
                
            return chunks
        except ImportError:
            raise ImportError("pandas is required for tabular data processing. Install with: pip install pandas openpyxl")