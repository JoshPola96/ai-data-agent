# app/services/ingest.py

"""
Document Ingestion Service
Production-grade parsing with universal data type handling and comprehensive logging
"""

import asyncio
import hashlib
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
from app.utils.helpers import to_numeric, totals_row_index

settings = get_settings()
logger = logging.getLogger(__name__)

# Every text-cleaning pass selects on this. Under pandas 3 a string column answers to
# "object" only for backward compatibility, and pandas 4 removes that — at which point
# selecting "object" alone would match nothing, and date detection, currency conversion
# and whitespace collapsing would each quietly become a no-op on the columns they exist
# to fix. Naming both dtypes costs nothing and survives the removal.
_TEXT_DTYPES = ["object", "string"]


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

        # Identity comes from the bytes, so the same file uploaded twice is the same
        # document. A random id made every upload new: two clicks left two registry
        # entries and two copies of every chunk competing for the same top-k slots, and
        # re-uploading a corrected file left the superseded version in the index forever,
        # answering from data the user believed they had replaced.
        doc_id = doc_id or self._content_doc_id(filename, content, file_path)
        logger.info(f"Document ID: {doc_id}")

        already = await self.session_store.get_session_documents(session_id)

        # The registry alone is not evidence the document is still searchable. The index
        # is discarded whenever it disagrees with its sidecar — a changed embedding model,
        # a torn write, a lost volume — while Redis survives, and every file then looked
        # "unchanged" and was skipped. The knowledge base came up empty and stayed empty,
        # because re-ingesting is exactly what the skip prevented.
        registered = any(d.get("doc_id") == doc_id for d in already)
        if registered and await self.vector_store.count_by_source(filename, session_id):
            logger.info(f"⏭ '{filename}' is byte-identical to a copy already here")
            return {
                "filename": filename,
                "doc_id": doc_id,
                "text_chunks": 0,
                "dataframes": 0,
                "status": "unchanged",
            }
        if registered:
            logger.warning(f"↻ '{filename}' is registered but not indexed, re-ingesting")

        try:
            # Parse file (pass both content and file_path)
            text_chunks, dataframes = await self._parse_file(
                filename, content, doc_id, file_path
            )

            logger.info(f"✓ Extracted {len(text_chunks)} text chunks")
            logger.info(f"✓ Extracted {len(dataframes)} tables")

            # Parsers return empty on failure rather than raising, so an unsupported or
            # corrupt file used to be reported as a successful upload with nothing in it.
            # The file then sat in the sidebar, unqueryable, looking ingested.
            if not text_chunks and not dataframes:
                raise ValueError(
                    f"Nothing could be extracted from '{filename}'. It may be corrupt, "
                    "empty, password-protected, or a scanned image with no text layer. "
                    "Supported formats: PDF, DOCX, XLSX, XLS, CSV, TXT."
                )

            # Only now is the previous version expendable. Superseding before parsing
            # meant re-uploading a corrupt copy of a good file deleted the good one and
            # then failed — the upload the user ran to refresh their data destroyed it.
            for old in [d for d in already if d.get("filename") == filename]:
                logger.info(f"♻ '{filename}' has changed, replacing the previous version")
                await self.session_store.delete_document(
                    session_id, old["doc_id"], self.vector_store
                )

            # Ownership stamp drives retrieval scoping and session deletion
            for chunk in text_chunks:
                chunk["session_id"] = session_id

            # Store vectors
            if text_chunks:
                logger.info("📚 Indexing text chunks in vector store...")
                await self.vector_store.add_documents(text_chunks)

            # Clean and store dataframes
            saved_tables = []
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
                saved_tables.append(name)

            # Register document in session
            await self.session_store.register_document(
                session_id,
                filename,
                doc_id,
                {
                    "text_chunks": len(text_chunks),
                    "dataframes": len(saved_tables),
                    # Recorded so deleting the document can delete its tables too
                    "tables": saved_tables,
                    "file_size": actual_file_size,
                    "has_structured_data": bool(saved_tables),
                    "file_type": filename.split(".")[-1].lower(),
                },
            )

            logger.info("=" * 80)
            logger.info(f"✅ INGESTION COMPLETE: {filename}")
            logger.info(f"   Text chunks: {len(text_chunks)}")
            logger.info(f"   Tables: {len(saved_tables)}")
            logger.info("=" * 80)

            return {
                "filename": filename,
                "doc_id": doc_id,
                "text_chunks": len(text_chunks),
                "dataframes": len(saved_tables),
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

        # 6. Drop a totals line loaded as data
        # Stating it in the profile was not enough: asked for a total, the agent read the
        # warning and answered 7,208 anyway — exactly double — treating the summary row as
        # a site called "Unknown". A row that is provably the sum of the others is an
        # artefact of the export, not an observation, so it is removed here and the removal
        # is reported rather than hidden.
        totals = totals_row_index(clean_df)
        if totals is not None:
            logger.info(f"   ➖ Dropped a totals row from '{table_name}' (row {totals})")
            clean_df = clean_df.drop(index=totals)

        # 7. Remove completely empty rows/columns
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
        Expose the number inside a text column, without consuming the text.

        A payslip exported as "Basic Pay 5000" hides a figure nothing can sum, so the
        number is lifted into its own column. It used to *replace* the original with the
        text part, which is indistinguishable from destroying the data: "Widget 500" and
        "Widget 750" both became "Widget", two products collapsed into one, and any
        grouping then double-counted them silently. "Report 2024" and "Report 2025" went
        the same way.

        There is no reliable signal separating a merged column from a product name —
        "Basic Pay 5000" and "Widget 500" are the same shape — so the split is additive.
        The original column is never touched, and the extracted number arrives beside it.
        Getting it wrong now costs a spare column instead of an identity.
        """
        # Pattern: Text followed by space and number
        merged_pattern = r"^(.+?)\s+([\d,]+(?:\.\d{1,2})?)$"

        new_columns = {}

        for col in df.select_dtypes(include=_TEXT_DTYPES).columns:
            # Check if column values match the pattern
            matches = df[col].astype(str).str.match(merged_pattern)
            match_rate = matches.mean()

            if match_rate > 0.4:  # 40% threshold
                logger.debug(
                    f"   🔧 Splitting merged column '{col}' ({match_rate:.1%} match rate)"
                )

                try:
                    extracted = df[col].astype(str).str.extract(merged_pattern)
                    values = pd.to_numeric(
                        extracted[1].str.replace(",", "", regex=False), errors="coerce"
                    )
                    # The original column stays exactly as it arrived; only the derived
                    # figure is added, and only when enough of it actually parsed.
                    if values.notna().mean() > 0.4:
                        new_columns[f"{col}_Value"] = values
                        logger.info(
                            f"   ＋ Derived '{col}_Value' from '{col}'; '{col}' unchanged"
                        )

                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to split column {col}: {e}")

        # Add new columns
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data

        return df

    def _detect_and_convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert date columns"""
        for col in df.select_dtypes(include=_TEXT_DTYPES).columns:
            # str(), because a column name is not always a string: a header row of years
            # loads as integers and `col.lower()` then raised AttributeError, taking the
            # whole file down rather than one column.
            if str(col).lower() in [x.lower() for x in settings.FORCE_STRING_COLUMNS]:
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

    @staticmethod
    def _is_identifier(name: str, series: pd.Series) -> bool:
        """
        True when a column labels rows rather than measuring them.

        Two independent signals, so neither has to be complete. The name is matched
        against FORCE_STRING_COLUMNS as whole `_`-separated tokens — "id" catches
        `customer_id` without "no" catching `notes` — and the values are checked for
        zero padding, because nothing meant to be added up is written `00067`. Converting
        anyway drops the padding and lets a customer number into the correlation matrix
        as though it were a measure.
        """
        lowered = str(name).lower().strip()
        parts = set(re.split(r"[^a-z0-9]+", lowered))
        if any(
            lowered == token.lower() or token.lower() in parts
            for token in settings.FORCE_STRING_COLUMNS
        ):
            return True

        text = series.dropna().astype(str).str.strip()
        return bool(len(text)) and text.str.fullmatch(r"0\d+").any()

    def _smart_numeric_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to numeric if appropriate"""
        for col in df.select_dtypes(include=_TEXT_DTYPES).columns:
            if self._is_identifier(col, df[col]):
                logger.debug(f"   ⊘ Keeping '{col}' as text (identifier, not a measure)")
                continue

            # One cleaner for the whole app. Stripping every non-digit character was
            # close but blunt: it read the European "1,5" as 15 and turned the
            # accounting negative "(300)" into 300.
            numeric_series = to_numeric(df[col])

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
        for col in df.select_dtypes(include=_TEXT_DTYPES).columns:
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
                    scan = pd.read_excel(
                        xls, sheet_name=sheet, dtype=str, header=None, nrows=8
                    )
                    start = self._header_row(scan)
                    df = pd.read_excel(
                        xls, sheet_name=sheet, dtype=str, header=start
                    )
                    if start:
                        logger.info(f"     ↧ '{sheet}': header found on row {start + 1}")

                    if not df.empty:
                        # Cleaned separately: the sheet name sits after the extension,
                        # and cleaning the pair as one string used to discard it.
                        tbl_name = (
                            f"{self._clean_name(filename)}_{self._clean_name(sheet)}"
                        )
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
            raw = open(file_path, "rb").read() if file_path else content

            # Encoding first, because getting it wrong rejects the whole file rather than
            # mangling a cell. Excel's "Unicode Text" export is UTF-16, and a European
            # Excel writes cp1252 — both raised UnicodeDecodeError and the upload was
            # refused as unreadable, which reads as "your file is broken".
            # UTF-16 is tried only behind its byte-order mark. Without one it is not
            # detectable — it decodes almost any even-length input into mojibake, which
            # swallowed a perfectly good cp1252 file and produced an empty table.
            ladder = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
            if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                ladder = ("utf-16",) + ladder

            text = None
            for encoding in ladder:
                try:
                    text = raw.decode(encoding)
                    if encoding not in ("utf-8-sig", "utf-8"):
                        logger.info(f"   ↻ Decoded as {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if text is None:
                raise ValueError("could not decode the file in any known encoding")

            # Then the delimiter, chosen by which one actually splits the file. Sniffing
            # with sep=None missed pipes on a short file and left every row as one string.
            df = None
            for sep in (",", ";", "\t", "|"):
                try:
                    candidate = pd.read_csv(io.StringIO(text), dtype=str, sep=sep)
                except Exception:
                    continue
                if df is None or candidate.shape[1] > df.shape[1]:
                    df = candidate
            if df is None:
                raise ValueError("no delimiter produced a readable table")

            logger.info(f"   Shape: {df.shape}")

            tbl_name = self._clean_name(filename)
            text_chunks = self._semantic_chunking(
                self._table_as_text(df, tbl_name), filename, doc_id
            )

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

    @staticmethod
    def _header_row(scan: pd.DataFrame) -> int:
        """
        Find the row carrying the column names.

        Exported reports open with a title and a "generated on" date, so the real header
        sits two or three rows down. Taken as-is, the title becomes the column names —
        `['Operations Export', 'Unnamed: 1', ...]` — the true header becomes the first
        row of data, and nothing in the sheet can be addressed by the name it actually
        has. The widest row wins, earliest on a tie, because a header names every column
        while a title occupies one cell.
        """
        if scan.empty:
            return 0

        filled = scan.notna().sum(axis=1)
        widest = int(filled.max())
        if widest < 2:
            return 0
        return int(filled[filled == widest].index[0])

    @staticmethod
    def _table_as_text(df: pd.DataFrame, label: str) -> str:
        """
        A searchable description of a table, not a transcript of it.

        Retrieval exists to find prose; the numbers are already queryable through the
        tools, and far more accurately. Indexing every row buys nothing and costs an
        embedding over the whole file.
        """
        preview = df.head(30).to_string()
        more = f"\n... and {len(df) - 30:,} further rows" if len(df) > 30 else ""
        columns = ", ".join(map(str, df.columns))
        return (
            f"Table {label}: {len(df):,} rows, {len(df.columns)} columns "
            f"({columns}).\n\n{preview}{more}"
        )

    def _semantic_chunking(
        self, text: str, source: str, doc_id: str, page_num: int = 1
    ) -> List[Dict]:
        """Smart semantic chunking that preserves context"""
        if not text or not text.strip():
            return []

        chunks = []

        # Splitting on blank lines alone leaves anything without them intact, however
        # long. A dataframe rendered to text has none, so a 6,000-row CSV arrived as a
        # single 258,000-character chunk — one embedding over a quarter of a megabyte and
        # a cross-encoder pass over the same, which took one retrieval from 30s to 8m22s.
        paragraphs = []
        for block in re.split(r"\n\s*\n", text):
            if len(block) <= settings.CHUNK_SIZE:
                paragraphs.append(block)
                continue
            step = max(1, settings.CHUNK_SIZE - settings.CHUNK_OVERLAP)
            paragraphs.extend(
                block[at : at + settings.CHUNK_SIZE] for at in range(0, len(block), step)
            )

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
        """
        Turn a filename or sheet name into an addressable table name.

        Only a trailing extension is removed. Splitting on the first dot dropped
        everything after it, so "book.xlsx_Sheet2" collapsed to "book" — every sheet in
        a workbook produced the same key and all but the last were silently overwritten.

        A name with no ASCII to keep falls back to a digest of the original: an
        Arabic-titled file sanitised down to the empty string, leaving a table nothing
        could refer to.
        """
        stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", stem).strip("_")
        if cleaned:
            return cleaned
        return f"table_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]}"

    def _content_doc_id(self, filename: str, content, file_path) -> str:
        """
        An id derived from the file's contents, so re-uploading is recognisable.

        Two uploads of the same bytes produce the same id and the second is skipped; an
        edited file produces a different one and supersedes its predecessor. Falls back to
        a random id only when there are no bytes to hash.
        """
        digest = hashlib.sha256()
        if file_path:
            with open(file_path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        elif content:
            digest.update(content)
        else:
            return self._generate_doc_id(filename)

        return f"{self._clean_name(filename)}_{digest.hexdigest()[:8]}"

    def _generate_doc_id(self, filename: str) -> str:
        """Generate unique document ID"""
        return f"{self._clean_name(filename)}_{uuid.uuid4().hex[:8]}"
