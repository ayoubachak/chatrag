import os
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
import tempfile
import uuid
from abc import ABC, abstractmethod
import re

class ChunkingStrategy(ABC):
    """
    Abstract base class for document chunking strategies.
    """
    
    @abstractmethod
    async def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Chunk a text document into smaller pieces.
        
        Args:
            text: The text to chunk
            metadata: Metadata about the source document
            
        Returns:
            List of document chunks with metadata
        """
        pass
        
    @abstractmethod
    async def chunk_pdf(self, pdf_reader, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk a PDF document into smaller pieces.
        
        Args:
            pdf_reader: The PDF reader object
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        pass
        
    @abstractmethod
    async def chunk_docx(self, doc, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk a Word document into smaller pieces.
        
        Args:
            doc: The Word document object
            file_path: Path to the Word file
            
        Returns:
            List of document chunks with metadata
        """
        pass
        
    @abstractmethod
    async def chunk_tabular(self, df, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk tabular data into smaller pieces.
        
        Args:
            df: The pandas DataFrame
            file_path: Path to the tabular file
            
        Returns:
            List of document chunks with metadata
        """
        pass

class BasicChunker(ChunkingStrategy):
    """
    Basic chunking strategy that splits text by paragraphs and processes
    documents page by page.
    """
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Chunk a text document by paragraphs.
        """
        # Simple chunking by paragraphs
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
    async def chunk_pdf(self, pdf_reader, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk a PDF document page by page.
        """
        text_chunks = []
        
        for i, page in enumerate(pdf_reader.pages):
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
        
    async def chunk_docx(self, doc, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk a Word document by paragraphs.
        """
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
        
    async def chunk_tabular(self, df, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk tabular data row by row.
        """
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

class SuperChunker(ChunkingStrategy):
    """
    Advanced chunking strategy that uses semantic boundaries and overlapping
    to create more meaningful chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the super chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    async def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Chunk a text document using semantic boundaries and overlapping.
        """
        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        # Create chunks with target size and overlap
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for i, para in enumerate(paragraphs):
            para_size = len(para)
            
            # If adding this paragraph would exceed the chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Find a good breaking point for the overlap
                # Start with the last self.chunk_overlap characters
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                
                # Try to find a sentence boundary in the overlap
                sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
                if len(sentences) > 1:
                    # Start the new chunk with the last sentence(s) from the previous chunk
                    current_chunk = sentences[-1]
                else:
                    # If no sentence boundary, just use the overlap
                    current_chunk = overlap_text
                
                current_size = len(current_chunk)
            
            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
                current_size += para_size + 2  # +2 for the newlines
            else:
                current_chunk = para
                current_size = para_size
                
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        # Create the final chunk objects with metadata
        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i,
                    "chunk_strategy": "semantic"
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
    async def chunk_pdf(self, pdf_reader, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk a PDF document using semantic boundaries across pages.
        """
        # First extract text from all pages
        all_text = ""
        page_breaks = []
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                all_text += page_text + "\n\n"
                page_breaks.append(len(all_text))
                
        # Use the text chunking method on the combined text
        base_metadata = {"source": Path(file_path).name}
        chunks = await self.chunk_text(all_text, base_metadata)
        
        # Add page information to each chunk
        for chunk in chunks:
            # Find which page(s) this chunk belongs to
            chunk_start = all_text.find(chunk["content"])
            chunk_end = chunk_start + len(chunk["content"])
            
            # Find the page number(s)
            start_page = 1
            end_page = 1
            
            for i, break_pos in enumerate(page_breaks):
                if chunk_start < break_pos:
                    start_page = i + 1
                    break
                    
            for i, break_pos in enumerate(page_breaks):
                if chunk_end <= break_pos:
                    end_page = i + 1
                    break
            
            # Update metadata
            chunk["metadata"]["page_start"] = start_page
            chunk["metadata"]["page_end"] = end_page
            
        return chunks
        
    async def chunk_docx(self, doc, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk a Word document using semantic boundaries.
        """
        # Extract all paragraphs
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        
        # Combine paragraphs into a single text
        all_text = "\n\n".join(paragraphs)
        
        # Use the text chunking method
        base_metadata = {"source": Path(file_path).name}
        return await self.chunk_text(all_text, base_metadata)
        
    async def chunk_tabular(self, df, file_path: str) -> List[Dict[str, str]]:
        """
        Chunk tabular data by groups of related rows.
        """
        chunks = []
        records = df.to_dict("records")
        
        # Group rows into chunks of appropriate size
        current_chunk = []
        current_size = 0
        
        for i, row in enumerate(records):
            # Format the row as a string
            row_text = "\n".join([f"{k}: {v}" for k, v in row.items()])
            row_size = len(row_text)
            
            # If adding this row would exceed the chunk size and we already have content,
            # save the current chunk and start a new one
            if current_size + row_size > self.chunk_size and current_chunk:
                # Combine the rows into a single chunk
                content = "\n\n".join(current_chunk)
                chunks.append({
                    "content": content,
                    "metadata": {
                        "source": Path(file_path).name,
                        "rows": f"{i-len(current_chunk)+1}-{i}",
                        "chunk_id": len(chunks),
                        "chunk_strategy": "semantic"
                    }
                })
                
                # Start a new chunk
                current_chunk = []
                current_size = 0
                
            # Add the row to the current chunk
            current_chunk.append(row_text)
            current_size += row_size
                
        # Add the last chunk if it's not empty
        if current_chunk:
            content = "\n\n".join(current_chunk)
            chunks.append({
                "content": content,
                "metadata": {
                    "source": Path(file_path).name,
                    "rows": f"{len(records)-len(current_chunk)+1}-{len(records)}",
                    "chunk_id": len(chunks),
                    "chunk_strategy": "semantic"
                }
            })
            
        return chunks

class DocumentLoader:
    """
    Utility class for loading and processing documents from various file formats.
    """
    
    def __init__(self, upload_dir: str = "uploads", chunking_strategy: Literal["basic", "super"] = "basic"):
        """
        Initialize the document loader.
        
        Args:
            upload_dir: Directory to store uploaded files
            chunking_strategy: Strategy to use for chunking documents
                - "basic": Simple chunking by paragraphs and pages
                - "super": Advanced semantic chunking with overlap
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Set the chunking strategy
        self.set_chunking_strategy(chunking_strategy)
        
    def set_chunking_strategy(self, strategy: Literal["basic", "super"]):
        """
        Set the chunking strategy.
        
        Args:
            strategy: The chunking strategy to use
        """
        if strategy == "super":
            self.chunker = SuperChunker()
        else:
            self.chunker = BasicChunker()
        
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
            
        # Use the chunking strategy
        metadata = {"source": Path(file_path).name}
        return await self.chunker.chunk_text(text, metadata)
        
    async def _load_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a PDF file.
        """
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            return await self.chunker.chunk_pdf(reader, file_path)
            
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: pip install pypdf")
            
    async def _load_docx(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a Word document.
        """
        try:
            import docx
            
            doc = docx.Document(file_path)
            return await self.chunker.chunk_docx(doc, file_path)
            
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
                
            return await self.chunker.chunk_tabular(df, file_path)
            
        except ImportError:
            raise ImportError("pandas is required for tabular data processing. Install with: pip install pandas openpyxl")