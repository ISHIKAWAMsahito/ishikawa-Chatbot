from google.generativeai.types import HarmCategory, HarmBlockThreshold
# パラメータ
PARAMS = {
    "QA_SIMILARITY_THRESHOLD": 0.90,
    "RERANK_SCORE_THRESHOLD": 6.0,
    "MAX_HISTORY_LENGTH": 20,
    "RERANK_TOP_K_INPUT": 15,# 検索上位15件をRerankにかける
}

# セーフティ設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

AI_MESSAGES = {
    "NOT_FOUND": (
        "申し訳ありません。ご質問に関連する確実な情報が資料内に見つかりませんでした。"
        "大学窓口へ直接お問い合わせいただくことをお勧めします。"
    ),
    "RATE_LIMIT": "申し訳ありません。現在アクセスが集中しています。1分ほど待ってから再度お試しください。",
    "SYSTEM_ERROR": "システムエラーが発生しました。しばらく時間をおいて再度お試しください。",
    "BLOCKED": "生成された回答がセーフティガイドラインに抵触したため、表示できませんでした。言い回しを変えて再度お試しください。"
}