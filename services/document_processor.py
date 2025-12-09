import io
import re
import logging
from typing import Iterator, Optional
import PyPDF2
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

class SimpleDocumentProcessor:
    """
    Render (メモリ512MB) 環境向け：メモリ効率重視の親子チャンキング対応プロセッサー
    
    特徴:
    1. 全データをリストで保持せず、Iterator (yield) で1件ずつ返す。
    2. 複雑なRetrieverを使わず、子チャンクのメタデータに「親の文脈」を埋め込む。
    """
    # ★ここが修正ポイント: 引数が parent_chunk_size, child_chunk_size になっています
    def __init__(self, parent_chunk_size=1500, child_chunk_size=400):
        # 親チャンク用スプリッター（文脈保持用：LLMに渡すテキスト）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=0, # 親同士の重複は基本不要
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        # 子チャンク用スプリッター（検索用ベクトル生成用）
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50, # 文脈の切れ目を防ぐため少しオーバーラップさせる
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        logging.info(f"SimpleDocumentProcessor Initialized: Parent={parent_chunk_size}, Child={child_chunk_size}")

    def _extract_text(self, filename: str, content: bytes) -> str:
        """ファイルタイプに応じてテキストを抽出する"""
        text = ""
        try:
            if filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    # ページ結合部に改行を入れる
                    text += (page.extract_text() or "") + "\n\n"
                logging.info(f".pdf extraction: {len(text)} chars (PyPDF2)")
            
            elif filename.endswith(".docx"):
                doc = DocxDocument(io.BytesIO(content))
                for para in doc.paragraphs:
                    text += para.text + "\n"
                logging.info(f".docx extraction: {len(text)} chars (python-docx)")
            
            elif filename.endswith(".txt"):
                text = content.decode('utf-8')
                logging.info(f".txt extraction: {len(text)} chars")
            
            else:
                logging.warning(f"Unsupported file type: {filename}")
                
            return self._clean_text(text)
        
        except Exception as e:
            logging.error(f"Text extraction error ({filename}): {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """テキストの整形（連続する空白の削除など）"""
        # 1. 連続する「空白・タブ・全角スペース」を1つの半角スペースに
        text = re.sub(r'[ \t\u3000]+', ' ', text)
        
        # 2. 連続する改行を最大2つまでに縮める（段落構造は維持）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def process_and_chunk(self, filename: str, content: bytes, category: str, collection_name: str) -> Iterator[LangChainDocument]:
        """
        ★重要: ListではなくIteratorを返す。
        処理したチャンクを即座にyieldし、メモリ上に巨大なリストを作らせない。
        """
        # 1. テキスト抽出
        full_text = self._extract_text(filename, content)
        if not full_text:
            return # 何も生成せず終了

        # 2. 親チャンク（大きな塊）を生成
        parent_chunks = self.parent_splitter.split_text(full_text)
        
        # full_textはメモリから解放可能な状態にする
        full_text = None 

        logging.info(f"{filename}: {len(parent_chunks)} parent chunks generated.")

        # 3. 各親チャンクをループ処理して子チャンクを作る
        for parent_idx, parent_text in enumerate(parent_chunks):
            
            # 親の中から子チャンク（小さな検索用ベクトル）を生成
            child_chunks = self.child_splitter.split_text(parent_text)
            
            for child_text in child_chunks:
                metadata = {
                    "source": filename,
                    "collection_name": collection_name,
                    "category": category,
                    "element_type": "ParentChild", 
                    "parent_index": parent_idx,
                    # ★重要: 検索ヒット時にLLMに渡すのは「親のテキスト」なので、
                    # ここでメタデータとして親テキストを持たせてしまう。
                    # これによりDB側での結合処理が不要になり、メモリ負荷も下がる。
                    "parent_context": parent_text 
                }
                
                # ドキュメントを1つ生成して即座に yield（呼び出し元に渡す）
                yield LangChainDocument(page_content=child_text, metadata=metadata)
            
        logging.info(f"{filename}: Processing complete (Stream finished).")

# グローバルインスタンス
simple_processor: Optional[SimpleDocumentProcessor] = None