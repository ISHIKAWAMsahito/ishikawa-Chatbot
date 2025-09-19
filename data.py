import chromadb

# --- ユーザーが設定する項目 ---

# データの追加先となるコレクションの名前を指定してください
# 例: "my_document_collection"
COLLECTION_NAME = "student-life"

# データベースファイルのパス
# このスクリプトと同じディレクトリに `chroma.sqlite3` を置いてください
DB_PATH = "."
DB_FILENAME = "chroma.sqlite3"

# ここに追加したいドキュメントのリストを記述してください
# 例: ["これは最初のドキュメントです。", "これは2番目の文章です。"]
documents_to_add = [
    "ここに1つ目のドキュメントやテキストを入力します。",
    "ここには2つ目のドキュメント。",
    "必要なだけドキュメントを追加できます。",
]

# 各ドキュメントに紐付ける一意のIDのリスト
# IDの数はドキュメントの数と一致させる必要があります
# 例: ["doc1", "doc2"]
ids_for_documents = [
    "id1", 
    "id2", 
    "id3"
]

# (任意) 各ドキュメントに紐付けるメタデータのリスト
# メタデータの数もドキュメントの数と一致させる必要があります
# 例: [{"source": "wiki"}, {"source": "news"}]
metadata_for_documents = [
    {"source": "user_input_1"},
    {"source": "user_input_2"},
    {"source": "user_input_3"},
]
# ------------------------------


def add_documents_to_chroma():
    """
    指定されたChromaDBのSQLiteファイルに接続し、ドキュメントを追加します。
    """
    try:
        # 永続的なストレージとしてローカルファイルを使用するクライアントを作成
        client = chromadb.PersistentClient(path=DB_PATH)
        print(f"データベース '{DB_FILENAME}' に接続しました。")

        # コレクションを取得または新規作成
        # get_or_create_collection を使うと、存在しない場合のみ作成されるので便利です
        print(f"コレクション '{COLLECTION_NAME}' を取得または作成します...")
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print("コレクションの準備ができました。")

        # ドキュメントの数とIDの数が一致しているかチェック
        if len(documents_to_add) != len(ids_for_documents):
            print("エラー: ドキュメントの数とIDの数が一致していません。")
            return
        
        if metadata_for_documents and len(documents_to_add) != len(metadata_for_documents):
            print("エラー: ドキュメントの数とメタデータの数が一致していません。")
            return

        # コレクションにドキュメントを追加
        print(f"{len(documents_to_add)}件のドキュメントを追加しています...")
        collection.add(
            documents=documents_to_add,
            metadatas=metadata_for_documents if metadata_for_documents else None,
            ids=ids_for_documents
        )
        
        # 追加後のコレクション内のアイテム数を確認
        count = collection.count()
        print("ドキュメントの追加が完了しました！")
        print(f"コレクション '{COLLECTION_NAME}' には現在 {count} 件のアイテムがあります。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("ChromaDBがインストールされているか確認してください (`pip install chromadb`)")

if __name__ == "__main__":
    if COLLECTION_NAME == "your_collection_name_here" or not documents_to_add or not ids_for_documents:
        print("スクリプトを実行する前に、`COLLECTION_NAME`, `documents_to_add`, `ids_for_documents` の値を設定してください。")
    else:
        add_documents_to_chroma()

