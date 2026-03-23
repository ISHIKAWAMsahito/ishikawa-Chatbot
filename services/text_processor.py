import html
import re

from models.schemas import InputValidationResult, OutputFilterResult
from services.constants import BLOCK_KEYWORDS, PII_PATTERNS


class TextProcessor:
    """テキスト処理を一元化するクラス。"""

    ROMAN_NUMERAL_PATTERN = r"^([Ⅰ-ⅫIIVX]+)[．\.\s]+(.*)$"

    @staticmethod
    def normalize_whitespace_and_newlines(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\t", " ").replace("　", " ")
        text = re.sub(r"(\n|\r|\r\n)+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def apply_markdown_headers(text: str) -> str:
        if not text:
            return ""
        return re.sub(
            TextProcessor.ROMAN_NUMERAL_PATTERN,
            r"# \1．\2",
            text,
            flags=re.MULTILINE,
        )

    @staticmethod
    def escape_html_content(text: str) -> str:
        if not text:
            return ""
        return html.escape(text, quote=True)

    @staticmethod
    def process_scraped_content(text: str, escape_html: bool = True) -> str:
        if not text:
            return ""
        text = TextProcessor.normalize_whitespace_and_newlines(text)
        if escape_html:
            text = TextProcessor.escape_html_content(text)
        return TextProcessor.apply_markdown_headers(text)


def inspect_input(text: str) -> InputValidationResult:
    """入力に危険キーワードが含まれるかを判定する。"""
    if not text:
        return InputValidationResult(is_safe=True, matched_keyword=None)

    lowered = text.lower()
    for keyword in BLOCK_KEYWORDS:
        if keyword in lowered:
            return InputValidationResult(is_safe=False, matched_keyword=keyword)
    return InputValidationResult(is_safe=True, matched_keyword=None)


def validate_input(text: str) -> bool:
    """危険入力でなければ True。"""
    return inspect_input(text).is_safe


def inspect_and_filter_output(text: str) -> OutputFilterResult:
    """PIIパターンを伏せ字にして結果を返す。"""
    if not text:
        return OutputFilterResult(filtered_text="", redaction_count=0)

    filtered = text
    redaction_count = 0
    for pattern in PII_PATTERNS:
        filtered, count = pattern.subn("****", filtered)
        redaction_count += count
    return OutputFilterResult(filtered_text=filtered, redaction_count=redaction_count)


def filter_output(text: str) -> str:
    """出力中の個人情報らしき文字列をマスクする。"""
    return inspect_and_filter_output(text).filtered_text