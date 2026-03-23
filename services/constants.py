import re
from typing import Pattern

# 入力検閲: SQL/OSコマンド注入を疑うキーワード（大文字小文字は text_processor 側で無視）
BLOCK_KEYWORDS: tuple[str, ...] = (
    "drop table",
    "delete from",
    "truncate table",
    "alter table",
    "union select",
    "xp_cmdshell",
    "system(",
    "rm -rf",
    "wget ",
    "curl ",
    "powershell ",
    "cmd.exe",
    "/bin/sh",
)

SAFETY_ERROR_MESSAGE = "安全上の理由により、その質問には回答できません。別の言い方で質問してください。"

# 出力フィルタ: 個人情報パターン
# 学籍番号（例: 英数字8桁）
STUDENT_ID_PATTERN: Pattern[str] = re.compile(r"\b[A-Za-z0-9]{8}\b")
# 電話番号（090-xxxx-xxxx / 080-xxxx-xxxx / 070-xxxx-xxxx）
PHONE_PATTERN: Pattern[str] = re.compile(r"\b0[789]0-\d{4}-\d{4}\b")

PII_PATTERNS: tuple[Pattern[str], ...] = (
    STUDENT_ID_PATTERN,
    PHONE_PATTERN,
)
