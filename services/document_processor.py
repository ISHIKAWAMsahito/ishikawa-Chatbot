import io
import os
import re
import hashlib
import logging
import tempfile
from typing import Any, Iterator, Optional

import pypdf
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document as LangChainDocument

class SimpleDocumentProcessor:
    """
    Render (メモリ512MB) 環境向け：メモリ効率重視の親子チャンキング対応プロセッサー
    特徴:
    1. DOCXなどの構造化ファイルをMarkdown形式に変換して抽出。
    2. 全データをリストで保持せず、Iterator (yield) で1件ずつ返す。
    3. 親子チャンキングのメタデータに「親の文脈」を埋め込む。
    """
    def __init__(self, parent_chunk_size=1500, child_chunk_size=400):
        # Markdown構造を意識したセパレーター設定
        # 見出し(###)や改行(\n\n)を優先して区切ることで、文脈の分断を防ぐ
        separators = [
            "\n#{1,6} ", # Markdownの見出し (# H1, ## H2...)
            "```\n",     # コードブロック
            "\n\n",      # 段落
            "\n",        # 行
            "。", "、", " ", ""
        ]

        # 親チャンク用スプリッター（文脈保持用）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
            separators=separators
        )
        # 子チャンク用スプリッター（検索用ベクトル生成用）
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50,
            separators=separators
        )
        logging.info(f"SimpleDocumentProcessor Initialized (Markdown Mode): Parent={parent_chunk_size}, Child={child_chunk_size}")

    def _extract_text_from_docx(self, content: bytes) -> str:
        """DOCXのスタイル情報を読み取り、Markdown形式のテキストに変換する"""
        try:
            doc = DocxDocument(io.BytesIO(content))
            full_text = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                style_name = para.style.name.lower()
                # スタイルをMarkdownにマッピング
                if 'heading 1' in style_name:
                    full_text.append(f"# {text}")
                elif 'heading 2' in style_name:
                    full_text.append(f"## {text}")
                elif 'heading 3' in style_name:
                    full_text.append(f"### {text}")
                elif 'list' in style_name or 'bullet' in style_name:
                    full_text.append(f"- {text}")
                else:
                    # 通常の段落
                    full_text.append(text)
            # 段落ごとに改行を2つ入れて結合
            return "\n\n".join(full_text)
        except Exception as e:
            logging.error(f"DOCX extraction error: {e}")
            return ""

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """PDFからテキスト抽出（PyPDF2使用、メモリ重視）"""
        text = ""
        try:
            # pypdfを使用するように修正
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
        except Exception as e:
            logging.error(f"PDF extraction error: {e}")
            return ""

    def _extract_text(self, filename: str, content: bytes) -> str:
        """ファイル拡張子に応じて適切な抽出メソッドを呼び出す"""
        text = ""
        try:
            if filename.endswith(".docx"):
                text = self._extract_text_from_docx(content)
                logging.info(f".docx parsed to Markdown: {len(text)} chars")
            elif filename.endswith(".pdf"):
                text = self._extract_text_from_pdf(content)
                # PDFはMarkdown変換が難しいため、基本的なクリーニングのみ
                text = self._clean_text(text)
                logging.info(f".pdf extracted: {len(text)} chars")
            elif filename.endswith(".txt") or filename.endswith(".md"):
                text = content.decode('utf-8')
                logging.info(f".txt/.md loaded: {len(text)} chars")
            else:
                logging.warning(f"Unsupported file type: {filename}")
            return text
        except Exception as e:
            logging.error(f"General text extraction error ({filename}): {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """汎用的なテキストクリーニング"""
        # 1. 連続する空白を1つに
        text = re.sub(r'[ \t\u3000]+', ' ', text)
        # 2. 3つ以上の連続改行を2つ（段落区切り）に統一
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process_and_chunk(self, filename: str, content: bytes, category: str, collection_name: str) -> Iterator[LangChainDocument]:
        """
        ListではなくIteratorを返す（メモリ節約）。
        DOCX等の場合はMarkdown形式でテキストが渡ってくるため、構造を維持したままチャンキングされる。
        """
        # 1. テキスト抽出 (DOCXならMarkdown化されている)
        full_text = self._extract_text(filename, content)
        if not full_text:
            return

        # 2. 親チャンク（大きな塊）を生成
        # Markdownヘッダーなどを考慮したスプリッター設定により、章の途中で切れにくくなる
        parent_chunks = self.parent_splitter.split_text(full_text)
        full_text = None # メモリ解放

        logging.info(f"{filename}: {len(parent_chunks)} parent chunks generated.")

        # 3. 各親チャンクをループ処理して子チャンクを作る
        for parent_idx, parent_text in enumerate(parent_chunks):
            # 親の中から子チャンクを生成
            child_chunks = self.child_splitter.split_text(parent_text)
            # document_processor.py の process_and_chunk メソッド内

            for child_text in child_chunks:
                metadata = {
                    "source": filename,
                    "collection_name": collection_name,
                    "category": category,
                    "element_type": "ParentChild",
                    "parent_index": parent_idx,
                    # ★修正: キー名を "parent_context" から "parent_content" に統一
                    # これにより、search.py や documents.py での扱いが統一されます
                    "parent_content": parent_text
                }
                yield LangChainDocument(page_content=child_text, metadata=metadata)
        logging.info(f"{filename}: Processing complete (Stream finished).")


# ---------------------------------------------------------------------------
# 高精度PDFモード: 1ページずつ画像化 → Storage へ直接アップロード → OCRスタブ → チャンキング
# （poppler_path は指定しない。Linux/Render では poppler-utils が PATH に必要）
# ---------------------------------------------------------------------------


def _ocr_stub_extract_text(jpeg_bytes: bytes, page_num: int, source_name: str) -> str:
    """
    OCR のスタブ: 本番では Tesseract / Vision API 等に差し替え可能。
    jpeg_bytes は将来の実OCRで使用する。
    """
    _ = jpeg_bytes
    return (
        f"[高精度モード ページ{page_num}] （OCRスタブ）\n"
        f"元ファイル: {source_name}\n"
        "※本番運用時はこのプレースホルダを実OCR結果に置き換えてください。"
    )


def _pdf_page_count(pdf_path: str) -> int:
    """pdf2image: 総ページ数（1ページ以上）。"""
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(pdf_path)
    raw = info.get("Pages", 1)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 1
    return max(1, n)


def _render_single_page_jpeg(pdf_path: str, page_num: int, dpi: int = 150) -> bytes:
    """1ページのみレンダリングし JPEG バイナリにする（全ページ一括禁止）。"""
    from pdf2image import convert_from_path

    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=dpi,
    )
    try:
        img = images[0]
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()
    finally:
        images.clear()
        del images


def _upload_jpeg_to_supabase(
    supabase_client: Any,
    bucket: str,
    storage_object_path: str,
    jpeg_bytes: bytes,
) -> None:
    """Supabase Storage に JPEG を直接アップロード（ローカルファイルに保存しない）。"""
    supabase_client.storage.from_(bucket).upload(
        storage_object_path,
        jpeg_bytes,
        {"content-type": "image/jpeg"},
    )


def iter_pdf_ocr_document_chunks(
    pdf_bytes: bytes,
    original_filename: str,
    category: str,
    collection_name: str,
    supabase_client: Any,
    processor: SimpleDocumentProcessor,
    bucket: str = "images",
    dpi: int = 150,
) -> Iterator[LangChainDocument]:
    """
    PDF を一時ファイルに保存し、ページ単位で画像化→Storage アップロード→OCRスタブ→SimpleDocumentProcessor でチャンキング。
    各チャンクの metadata に element_type=image_ocr, image_path, source を付与する。
    """
    safe_base = re.sub(r"[^\w\.\-]", "_", os.path.basename(original_filename))[:120] or "document"
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(fd, pdf_bytes)
        os.close(fd)
        n_pages = _pdf_page_count(path)
        for page_num in range(1, n_pages + 1):
            jpeg_bytes = _render_single_page_jpeg(path, page_num, dpi=dpi)
            digest = hashlib.sha256(jpeg_bytes).hexdigest()[:24]
            image_filename = f"{safe_base}_p{page_num}_{digest}.jpg"
            storage_object_path = f"pdf_ocr/{image_filename}"
            _upload_jpeg_to_supabase(
                supabase_client, bucket, storage_object_path, jpeg_bytes
            )
            ocr_text = _ocr_stub_extract_text(jpeg_bytes, page_num, original_filename)
            synthetic_filename = f"{safe_base}_p{page_num}.txt"
            for doc in processor.process_and_chunk(
                synthetic_filename,
                ocr_text.encode("utf-8"),
                category,
                collection_name,
            ):
                doc.metadata["element_type"] = "image_ocr"
                doc.metadata["image_path"] = image_filename
                doc.metadata["source"] = original_filename
                yield doc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# グローバルインスタンス
simple_processor: Optional[SimpleDocumentProcessor] = None