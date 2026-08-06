# app/services/ingest.py

"""
Document Ingestion Service
Production-grade parsing with universal data type handling and comprehensive logging
"""

import asyncio
import io
import logging
import os
import re
import uuid
from typing import List, Dict, Tuple
import pandas as pd
import pdfplumber
import docx
import numpy as np
from app.core.config import get_settings
from app.core.session import SessionStore
from app.core.vectorstore import VectorStore

settings = get_settings()
logger = logging.getLogger(__name__)


class IngestionService:
    """
    Production-grade file ingestion with:
    - Visual PDF table extraction
    - Universal data type handling
    - Automatic data cleaning and normalization
    - Comprehensive error tracking
    """

    def __init__(self, vector_store: VectorStore, session_store: SessionStore):
        self.vector_store = vector_store
        self.session_store = session_store

    async def ingest_file(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        doc_id: str = None,
        file_path: str = None,
    ) -> Dict:
        """Main entry point with granular error tracking."""
        logger.info("=" * 80)
        logger.info(f"📥 INGESTING FILE: {filename}")
        logger.info("=" * 80)
        logger.info(f"Session: {session_id[:8]}")

        # Calculate size safely based on input type
        if file_path:
            actual_file_size = os.path.getsize(file_path)
        elif content:
            actual_file_size = len(content)
        else:
            actual_file_size = 0

        logger.info(f"Size: {actual_file_size:,} bytes")

        doc_id = doc_id or self._generate_doc_id(filename)
        logger.info(f"Document ID: {doc_id}")

        try:
            # Parse file (pass both content and file_path)
            text_chunks, dataframes = await self._parse_file(
                filename, content, doc_id, file_path
            )

            logger.info(f"✓ Extracted {len(text_chunks)} text chunks")
            logger.info(f"✓ Extracted {len(dataframes)} tables")

            # Ownership stamp drives retrieval scoping and session deletion
            for chunk in text_chunks:
                chunk["session_id"] = session_id

            # Store vectors
            if text_chunks:
                logger.info("📚 Indexing text chunks in vector store...")
                await self.vector_store.add_documents(text_chunks)

            # Clean and store dataframes
            saved_tables = 0
            for name, df in dataframes.items():
                logger.info(
                    f"📊 Processing table '{name}' ({df.shape[0]}x{df.shape[1]})"
                )

                # Apply universal data cleaning
                clean_df = await self._universal_data_cleaning(df, name)

                logger.debug(f"   Cleaned shape: {clean_df.shape}")
                logger.debug(f"   Columns: {list(clean_df.columns)}")
                logger.debug(f"   Dtypes: {dict(clean_df.dtypes)}")

                await self.session_store.save_dataframe(session_id, name, clean_df)
                saved_tables += 1

            # Register document in session
            await self.session_store.register_document(
                session_id,
                filename,
                doc_id,
                {
                    "text_chunks": len(text_chunks),
                    "dataframes": saved_tables,
                    "file_size": actual_file_size,
                    "has_structured_data": saved_tables > 0,
                    "file_type": filename.split(".")[-1].lower(),
                },
            )

            logger.info("=" * 80)
            logger.info(f"✅ INGESTION COMPLETE: {filename}")
            logger.info(f"   Text chunks: {len(text_chunks)}")
            logger.info(f"   Tables: {saved_tables}")
            logger.info("=" * 80)

            return {
                "filename": filename,
                "doc_id": doc_id,
                "text_chunks": len(text_chunks),
                "dataframes": saved_tables,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"❌ Ingestion failed for {filename}: {e}", exc_info=True)
            raise

    async def _parse_file(
        self, filename: str, content: bytes, doc_id: str, file_path: str = None
    ) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Route to specialized parsers."""
        ext = filename.lower().split(".")[-1]

        logger.debug(f"🔍 File extension: {ext}")

        parsers = {
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "xlsx": self._parse_excel,
            "xls": self._parse_excel,
            "csv": self._parse_csv,
            "txt": self._parse_text,
        }

        parser = parsers.get(ext)
        if not parser:
            logger.warning(f"⚠️ Unsupported format: {ext}")
            return [], {}

        logger.info(f"📄 Using parser: {parser.__name__}")

        # Offload parsing to thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, parser, filename, content, doc_id, file_path
        )

    # =========================================================================
    # UNIVERSAL DATA CLEANING
    # =========================================================================

    async def _universal_data_cleaning(
        self, df: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        """
        Universal data type handling and cleaning.

        Strategy:
        1. Handle duplicate columns
        2. Split merged columns (e.g., "Description 5000" -> "Description", "5000")
        3. Detect and convert dates
        4. Smart numeric conversion
        5. Clean string data
        """
        logger.debug(f"🧹 Cleaning dataframe '{table_name}'")
        logger.debug(f"   Input shape: {df.shape}")
        logger.debug(f"   Input columns: {list(df.columns)}")

        # Make a copy
        clean_df = df.copy()

        # 1. Handle duplicate columns
        clean_df = self._fix_duplicate_columns(clean_df)

        # 2. Split merged columns
        clean_df = self._split_merged_columns(clean_df)

        # 3. Detect and convert dates
        clean_df = self._detect_and_convert_dates(clean_df)

        # 4. Smart numeric conversion
        clean_df = self._smart_numeric_conversion(clean_df)

        # 5. Clean string columns
        clean_df = self._clean_string_columns(clean_df)

        # 6. Remove completely empty rows/columns
        clean_df = clean_df.dropna(how="all").dropna(axis=1, how="all")

        logger.debug(f"   Output shape: {clean_df.shape}")
        logger.debug(f"   Output dtypes: {dict(clean_df.dtypes)}")

        return clean_df

    def _fix_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate column names"""
        if df.columns.duplicated().any():
            logger.warning("   ⚠️ Found duplicate columns, deduplicating...")
            # Keep only the first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _split_merged_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split columns that contain both text and numbers.
        Example: "Basic Pay 5000" -> "Basic Pay" | "5000"
        """
        # Pattern: Text followed by space and number
        merged_pattern = r"^(.+?)\s+([\d,]+(?:\.\d{1,2})?)$"

        new_columns = {}

        for col in df.select_dtypes(include=["object"]).columns:
            # Check if column values match the pattern
            matches = df[col].astype(str).str.match(merged_pattern)
            match_rate = matches.mean()

            if match_rate > 0.4:  # 40% threshold
                logger.debug(
                    f"   🔧 Splitting merged column '{col}' ({match_rate:.1%} match rate)"
                )

                try:
                    extracted = df[col].astype(str).str.extract(merged_pattern)

                    # Update original to text part
                    df[col] = extracted[0].str.strip()

                    # Create new numeric column
                    new_col_name = f"{col}_Value"
                    new_columns[new_col_name] = pd.to_numeric(
                        extracted[1].str.replace(",", "", regex=False), errors="coerce"
                    )

                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to split column {col}: {e}")

        # Add new columns
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data

        return df

    def _detect_and_convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert date columns"""
        for col in df.select_dtypes(include=["object"]).columns:
            # Skip if column is in force string list
            if col.lower() in [x.lower() for x in settings.FORCE_STRING_COLUMNS]:
                continue

            # Try date conversion
            sample = df[col].dropna().head(10)
            if len(sample) == 0:
                continue

            # Try multiple date formats
            for date_format in settings.DATE_FORMATS:
                try:
                    test_convert = pd.to_datetime(
                        sample, format=date_format, errors="coerce"
                    )

                    # If >70% successful, convert entire column
                    if test_convert.notna().mean() > 0.7:
                        df[col] = pd.to_datetime(
                            df[col], format=date_format, errors="coerce"
                        )
                        logger.debug(
                            f"   📅 Converted '{col}' to datetime (format: {date_format})"
                        )
                        break

                except Exception:
                    continue

        return df

    def _smart_numeric_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to numeric if appropriate"""
        for col in df.select_dtypes(include=["object"]).columns:
            # Skip if in force string list
            if col.lower() in [x.lower() for x in settings.FORCE_STRING_COLUMNS]:
                logger.debug(
                    f"   ⊘ Skipping numeric conversion for '{col}' (force string)"
                )
                continue

            # Clean and try conversion
            clean_series = df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            numeric_series = pd.to_numeric(clean_series, errors="coerce")

            # Convert if enough values are valid numbers
            valid_rate = numeric_series.notna().mean()
            if valid_rate >= settings.NUMERIC_CONVERSION_THRESHOLD:
                df[col] = numeric_series
                logger.debug(
                    f"   🔢 Converted '{col}' to numeric ({valid_rate:.1%} valid)"
                )

        return df

    def _clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string columns"""
        for col in df.select_dtypes(include=["object"]).columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()

            # Replace 'nan', 'None', etc. with actual NaN
            df[col] = df[col].replace(["nan", "None", "NULL", ""], np.nan)

        return df

    # =========================================================================
    # PDF PARSER
    # =========================================================================

    def _parse_pdf(
        self, filename: str, content: bytes, doc_id: str, file_path: str = None
    ) -> Tuple[List[Dict], Dict]:
        """Parse PDF with advanced table extraction"""
        logger.info(f"📄 Parsing PDF: {filename}")

        text_chunks = []
        dataframes = {}
        total_pages = 0

        # Use file path directly if available
        target = file_path if file_path else io.BytesIO(content)

        try:
            with pdfplumber.open(target) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"   Total pages: {total_pages}")

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    logger.debug(f"   Processing page {page_num}/{total_pages}")

                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        logger.debug(
                            f"     Found {len(tables)} tables on page {page_num}"
                        )

                    for t_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue

                        try:
                            # Create DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df = df.dropna(how="all").dropna(axis=1, how="all")

                            if not df.empty:
                                tbl_name = self._clean_name(
                                    f"{filename}_p{page_num}_t{t_idx + 1}"
                                )
                                dataframes[tbl_name] = df

                                logger.debug(f"     ✓ Table '{tbl_name}': {df.shape}")

                                # Add table summary for RAG
                                table_md = df.head(5).to_markdown(index=False)
                                text_chunks.append(
                                    {
                                        "content": f"Table '{tbl_name}' on Page {page_num}:\n{table_md}",
                                        "source": filename,
                                        "doc_id": doc_id,
                                        "metadata": {
                                            "page": page_num,
                                            "type": "table",
                                            "table_name": tbl_name,
                                        },
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"     ⚠️ Failed to parse table: {e}")

                    # Extract text
                    text = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if text:
                        page_chunks = self._semantic_chunking(
                            text, filename, doc_id, page_num
                        )
                        text_chunks.extend(page_chunks)
                        logger.debug(f"     ✓ Extracted {len(page_chunks)} text chunks")

            logger.info(
                f"✅ PDF parsed: {len(text_chunks)} chunks, {len(dataframes)} tables"
            )
            return text_chunks, dataframes

        except Exception as e:
            logger.error(f"❌ PDF parsing error: {e}", exc_info=True)
            return [], {}

    # =========================================================================
    # OTHER PARSERS
    # =========================================================================

    def _parse_docx(
        self, filename: str, content: bytes, doc_id: str, file_path: str = None
    ) -> Tuple[List[Dict], Dict]:
        """Parse DOCX files"""
        logger.info(f"📄 Parsing DOCX: {filename}")

        try:
            # Use file path directly
            target = file_path if file_path else io.BytesIO(content)
            doc = docx.Document(target)

            # Extract text
            full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            logger.debug(f"   Extracted {len(full_text)} characters of text")

            # Extract tables
            dataframes = {}
            for i, table in enumerate(doc.tables):
                data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                if len(data) > 1:
                    try:
                        df = pd.DataFrame(data[1:], columns=data[0])
                        df = df.dropna(how="all").dropna(axis=1, how="all")
                        if not df.empty:
                            tbl_name = self._clean_name(f"{filename}_tbl{i + 1}")
                            dataframes[tbl_name] = df
                            logger.debug(f"   ✓ Table '{tbl_name}': {df.shape}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to parse table {i}: {e}")

            text_chunks = self._semantic_chunking(full_text, filename, doc_id)

            logger.info(
                f"✅ DOCX parsed: {len(text_chunks)} chunks, {len(dataframes)} tables"
            )
            return text_chunks, dataframes

        except Exception as e:
            logger.error(f"❌ DOCX parsing error: {e}", exc_info=True)
            return [], {}

    def _parse_excel(
        self, filename: str, content: bytes, doc_id: str, file_path: str = None
    ) -> Tuple[List[Dict], Dict]:
        """Parse Excel files"""
        logger.info(f"📊 Parsing Excel: {filename}")

        try:
            # Use file path directly
            target = file_path if file_path else io.BytesIO(content)
            xls = pd.ExcelFile(target)
            logger.info(f"   Found {len(xls.sheet_names)} sheets")

            dataframes = {}
            text_summary = []

            for sheet in xls.sheet_names:
                logger.debug(f"   Processing sheet: {sheet}")

                try:
                    df = pd.read_excel(xls, sheet_name=sheet, dtype=str)

                    if not df.empty:
                        tbl_name = self._clean_name(f"{filename}_{sheet}")
                        dataframes[tbl_name] = df

                        logger.debug(f"     ✓ {tbl_name}: {df.shape}")

                        # Add summary
                        text_summary.append(
                            f"Sheet '{sheet}' Preview:\n{df.head(5).to_string()}"
                        )
                except Exception as e:
                    logger.warning(f"     ⚠️ Failed to parse sheet '{sheet}': {e}")

            full_text = "\n\n".join(text_summary)
            text_chunks = self._semantic_chunking(full_text, filename, doc_id)

            logger.info(
                f"✅ Excel parsed: {len(text_chunks)} chunks, {len(dataframes)} tables"
            )
            return text_chunks, dataframes

        except Exception as e:
            logger.error(f"❌ Excel parsing error: {e}", exc_info=True)
            return [], {}

    def _parse_csv(
        self, filename: str, content: bytes, doc_id: str, file_path: str = None
    ) -> Tuple[List[Dict], Dict]:
        """Parse CSV files"""
        logger.info(f"📊 Parsing CSV: {filename}")

        try:
            # Use file path directly
            target = file_path if file_path else io.BytesIO(content)
            df = pd.read_csv(target, dtype=str)
            logger.info(f"   Shape: {df.shape}")

            tbl_name = self._clean_name(filename)
            text_chunks = self._semantic_chunking(df.to_string(), filename, doc_id)

            logger.info(f"✅ CSV parsed: {len(text_chunks)} chunks, 1 table")
            return text_chunks, {tbl_name: df}

        except Exception as e:
            logger.error(f"❌ CSV parsing error: {e}", exc_info=True)
            return [], {}

    def _parse_text(
        self, filename: str, content: bytes, doc_id: str, file_path: str = None
    ) -> Tuple[List[Dict], Dict]:
        """Parse text files"""
        logger.info(f"📄 Parsing text file: {filename}")

        try:
            # Read from file if available
            if file_path:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = content.decode("utf-8", errors="ignore")

            logger.info(f"   Length: {len(text)} characters")

            text_chunks = self._semantic_chunking(text, filename, doc_id)

            logger.info(f"✅ Text parsed: {len(text_chunks)} chunks")
            return text_chunks, {}

        except Exception as e:
            logger.error(f"❌ Text parsing error: {e}", exc_info=True)
            return [], {}

    # =========================================================================
    # CHUNKING
    # =========================================================================

    def _semantic_chunking(
        self, text: str, source: str, doc_id: str, page_num: int = 1
    ) -> List[Dict]:
        """Smart semantic chunking that preserves context"""
        if not text or not text.strip():
            return []

        chunks = []
        paragraphs = re.split(r"\n\s*\n", text)

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding exceeds limit, save current
            if len(current_chunk) + len(para) > settings.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(
                        self._create_chunk_obj(current_chunk, source, doc_id, page_num)
                    )
                    tail = self._overlap_tail(current_chunk)
                    current_chunk = f"{tail}\n\n{para}" if tail else para
                else:
                    current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

        # Keep a short trailing remainder only if it would otherwise be the whole document
        if current_chunk and (
            len(current_chunk) >= settings.MIN_CHUNK_SIZE or not chunks
        ):
            chunks.append(
                self._create_chunk_obj(current_chunk, source, doc_id, page_num)
            )

        return chunks

    def _overlap_tail(self, chunk: str) -> str:
        """Trailing slice carried into the next chunk so context survives the split."""
        if settings.CHUNK_OVERLAP <= 0:
            return ""

        tail = chunk[-settings.CHUNK_OVERLAP :]
        # Start at a word boundary rather than mid-token
        _, sep, rest = tail.partition(" ")
        return (rest if sep else tail).strip()

    def _create_chunk_obj(self, text, source, doc_id, page):
        """Create chunk object"""
        return {
            "content": text,
            "source": source,
            "doc_id": doc_id,
            "metadata": {"page": page, "chunk_size": len(text), "type": "text"},
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _clean_name(self, name: str) -> str:
        """Clean table/file names"""
        name = name.split(".")[0]
        return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")

    def _generate_doc_id(self, filename: str) -> str:
        """Generate unique document ID"""
        return f"{self._clean_name(filename)}_{uuid.uuid4().hex[:8]}"
