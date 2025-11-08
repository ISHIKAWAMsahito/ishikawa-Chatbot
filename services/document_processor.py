import io
import re
import logging
from typing import List, Optional
import PyPDF2
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

class SimpleDocumentProcessor:
    """
    メモリを消費しない、単純なテキスト抽出とチャンキングを行うクラス。
    unstructured や 親子チャンキング は使用しない。
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "、", " "]
        )
        logging.info(f"SimpleDocumentProcessor (Chunk: {chunk_size}/{chunk_overlap}) が初期化されました。")

    def _extract_text(self, filename: str, content: bytes) -> str:
        """ファイルタイプに応じてテキストを抽出する"""
        text = ""
        try:
            if filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    text += page.extract_text()
                logging.info(f".pdf から {len(text)} 文字を抽出 (PyPDF2)")
            
            elif filename.endswith(".docx"):
                doc = DocxDocument(io.BytesIO(content))
                for para in doc.paragraphs:
                    text += para.text + "\n"
                logging.info(f".docx から {len(text)} 文字を抽出 (python-docx)")
            
            elif filename.endswith(".txt"):
                text = content.decode('utf-8')
                logging.info(f".txt から {len(text)} 文字を抽出")
            
            else:
                logging.warning(f"未対応のファイル形式: {filename}")
                
            return re.sub(r'\s+', ' ', text).strip()
        
        except Exception as e:
            logging.error(f"テキスト抽出エラー ({filename}): {e}")
            return ""

    def process_and_chunk(self, filename: str, content: bytes, category: str, collection_name: str) -> List[LangChainDocument]:
        """
        1. テキストを抽出
        2. 1000/200 でチャンキング
        3. メタデータを付与
        """
        # 1. テキスト抽出
        full_text = self._extract_text(filename, content)
        if not full_text:
            return []

        # 2. チャンキング
        chunks = self.splitter.split_text(full_text)
        
        # 3. メタデータ付与
        docs = []
        for chunk_text in chunks:
            metadata = {
                "source": filename,
                "collection_name": collection_name,
                "category": category,
                "element_type": "SimpleChunk",
            }
            docs.append(LangChainDocument(page_content=chunk_text, metadata=metadata))
            
        logging.info(f"{filename} から {len(docs)} 件の単純チャンクを生成しました。")
        return docs

# グローバルインスタンス
simple_processor: Optional[SimpleDocumentProcessor] = None