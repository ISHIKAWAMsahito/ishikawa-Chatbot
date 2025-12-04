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
    """
    # 修正推奨: chunk_size を 1000 から 1500 または 2000 に増やす
    # gemini-embedding-001 は入力制限に余裕があるため、長めのチャンクでも処理可能です。
    def __init__(self, chunk_size=1500, chunk_overlap=300): 
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        logging.info(f"SimpleDocumentProcessor (Chunk: {chunk_size}/{chunk_overlap}) が初期化されました。")

    def _extract_text(self, filename: str, content: bytes) -> str:
        """ファイルタイプに応じてテキストを抽出する"""
        text = ""
        try:
            if filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # ★修正: ページごとの結合部分に改行を入れる
                        text += page_text + "\n\n"
                logging.info(f".pdf から {len(text)} 文字を抽出 (PyPDF2)")
            
            elif filename.endswith(".docx"):
                doc = DocxDocument(io.BytesIO(content))
                for para in doc.paragraphs:
                    # docxはパラグラフごとに改行が入るのでそのままでOK
                    text += para.text + "\n"
                logging.info(f".docx から {len(text)} 文字を抽出 (python-docx)")
            
            elif filename.endswith(".txt"):
                text = content.decode('utf-8')
                logging.info(f".txt から {len(text)} 文字を抽出")
            
            else:
                logging.warning(f"未対応のファイル形式: {filename}")
                
            return self._clean_text(text)
        
        except Exception as e:
            logging.error(f"テキスト抽出エラー ({filename}): {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """
        ★修正: 改行を完全に消さず、整形する
        """
        # 1. 連続する「空白・タブ」を1つのスペースに (改行は含めない)
        text = re.sub(r'[ \t\u3000]+', ' ', text)
        
        # 2. 連続する改行を最大2つまでにする（3つ以上あれば2つに縮める）
        #    これにより「段落」の意味を残せる
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. 前後の空白削除
        return text.strip()

    def process_and_chunk(self, filename: str, content: bytes, category: str, collection_name: str) -> List[LangChainDocument]:
        """
        1. テキストを抽出
        2. チャンキング
        3. メタデータを付与
        """
        # 1. テキスト抽出
        full_text = self._extract_text(filename, content)
        if not full_text:
            return []

        # 2. チャンキング
        # 改行が残っているので、セパレーターが正しく機能する
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