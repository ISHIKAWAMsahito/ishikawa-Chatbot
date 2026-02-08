import html
import re
from typing import Optional

class TextProcessor:
    """テキスト処理を一元化するクラス
    
    ホワイトスペース正規化、マークダウン変換、HTMLエスケープを統一的に処理します
    """
    
    # マークダウンヘッダーのマッピング
    ROMAN_NUMERAL_PATTERN = r'^
    
    @staticmethod
    def normalize_whitespace_and_newlines(text: str) -> str:
        """ホワイトスペースと改行を正規化する
        
        Args:
            text: 正規化対象のテキスト
            
        Returns:
            正規化されたテキスト
        """
        if not text:
            return ""
        
        # タブと全角スペースを半角スペースに統一
        text = text.replace("\t", " ").replace("　", " ")
        
        # 複数の改行をシングル改行に統一
        text = re.sub(r'(\n|\r|\r\n)+', '\n', text)
        
        # 複数の連続スペースをシングルスペースに統一
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def apply_markdown_headers(text: str) -> str:
        """ローマ数字の見出しをマークダウンヘッダーに変換する
        
        Args:
            text: 変換対象のテキスト
            
        Returns:
            マークダウン適用後のテキスト
        """
        if not text:
            return ""
        
        # ローマ数字見出しをマークダウンH1に変換
        text = re.sub(
            TextProcessor.ROMAN_NUMERAL_PATTERN,
            r'# \1．\2',
            text,
            flags=re.MULTILINE
        )
        
        return text
    
    @staticmethod
    def escape_html_content(text: str) -> str:
        """HTMLコンテンツをエスケープして XSS を防ぐ
        
        Args:
            text: エスケープ対象のテキスト
            
        Returns:
            エスケープされたテキスト
        """
        if not text:
            return ""
        
        return html.escape(text, quote=True)
    
    @staticmethod
    def process_scraped_content(text: str, escape_html: bool = True) -> str:
        """Webスクレイプコンテンツを処理する
        
        Args:
            text: 処理対象のテキスト
            escape_html: HTMLエスケープを実行するか（デフォルト: True）
            
        Returns:
            処理されたテキスト
        """
        if not text:
            return ""
        
        # ステップ1: ホワイトスペース正規化
        text = TextProcessor.normalize_whitespace_and_newlines(text)
        
        # ステップ2: HTMLエスケープ（スクレイプコンテンツのセキュリティ対策）
        if escape_html:
            text = TextProcessor.escape_html_content(text)
        
        # ステップ3: マークダウン適用
        text = TextProcessor.apply_markdown_headers(text)
        
        return text