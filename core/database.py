import logging#Python標準のログ出力機能を読み込んでいます。エラー発生時などに記録を残すために使います
from typing import List, Optional, Dict, Any#型ヒント（Type Hinting）を使うための部品を読み込んでいます。
# List: リスト型（配列）、Optional: 値がNoneになる可能性があるもの、Dict: 辞書型、Any: どんな型でもOK、という意味です 。
from supabase import create_client, Client#Supabase公式ライブラリから、接続クライアントを作る関数 create_client と、クライアントの型定義 Client を読み込んでいます

class SupabaseClientManager:#Supabaseに関する操作をまとめて管理するクラス（設計図）の定義開始です 。
    """Supabaseクライアント管理クラス"""
    def __init__(self, url: str, key: str):#このクラスを使うとき最初に実行される初期化メソッドです。引数としてSupabaseの url とAPI key を受け取ります 。
        self.client: Client = create_client(url, key)#受け取ったURLとキーを使ってSupabaseに接続し、その接続情報（クライアントインスタンス）を self.client という変数に保存します。以後、この変数を使ってDB操作を行います 。

    def get_db_type(self) -> str:#現在使っているデータベースの種類を文字列で返すメソッドです 。
        return "supabase"#固定で "supabase" という文字を返しています。将来的に他のDBに切り替える場合、ここを変えることでシステム側で判別できるようにするための設計と思われます 。

    def insert_document(self, content: str, embedding: List[float], metadata: dict):#ドキュメントをDBに登録するメソッドです。本文(content)、ベクトルデータ(embedding)、付加情報(metadata)を受け取ります 。
        self.client.table("documents").insert({#table("documents"): Supabase上の "documents" テーブルを指定します。
    #insert({...}): 辞書形式で渡されたデータを挿入する準備をします。
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }).execute()#execute(): 実際にDBへの書き込み命令を実行します 。

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:#検索条件（コレクション名、カテゴリ、検索用ベクトル、取得件数）を受け取り、結果を辞書のリストで返すメソッドです 。
        params = {#Supabase側の関数に渡すためのパラメータを辞書にまとめています。キー名（p_collection_nameなど）はSupabase側のSQL関数の引数名と一致させる必要があります 。
            "p_collection_name": collection_name,
            "p_category": category,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents", params).execute()#rpc(...): Remote Procedure Callの略。Supabase側にあらかじめ定義した
        # match_documents という関数を呼び出しています。これがベクトル検索の実体です 。
        return result.data or []#return result.data or []:検索結果(result.data)があればそれを返し、もし空(None)なら空リスト [] を返します
    def search_documents_hybrid(self, collection_name: str, query_text: str, query_embedding: List[float], match_count: int) -> List[dict]:
        """
        キーワード検索とベクトル検索を組み合わせたハイブリッド検索を実行する
        """
        params = {
            "p_collection_name": collection_name,
            "p_query_text": query_text,       # ユーザーの生の質問文
            "p_query_embedding": query_embedding, # Embeddingされたベクトル
            "p_match_count": match_count
        }
        # Supabaseのmatch_documents_hybrid関数をRPCで呼び出す
        result = self.client.rpc("match_documents_hybrid", params).execute()
        return result.data or []
    def search_documents_by_vector(self, collection_name: str, embedding: List[float], match_count: int) -> List[dict]:#カテゴリ引数(category)が無いバージョンの検索メソッドです 。
    # パラメータから p_category がなくなっています 。
        """カテゴリで絞り込まずにベクトル検索を行う"""
        params = {
            "p_collection_name": collection_name,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents_by_vector", params).execute()#result = self.client.rpc("match_documents_by_vector", params).execute():Supabase側の別の関数 match_documents_by_vector を呼び出しています。これはカテゴリフィルタを行わない検索用です 。
        return result.data or []

    def search_fallback_qa(self, embedding: List[float], match_count: int) -> List[dict]:#ドキュメントではなく、Q&Aデータから検索するためのメソッドです 。
        """Q&Aフォールバックをベクトル検索する"""
        params = {
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        result = self.client.rpc("match_fallback_qa", params).execute()#result = self.client.rpc("match_fallback_qa", params).execute():match_fallback_qa というQ&A検索専用のSQL関数を呼び出しています 。
        return result.data or []

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        result = self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute()#select("id, metadata"):全項目ではなく、管理に必要な id と metadata だけを取得しています（通信量を減らすため） 。
        #eq("metadata->>collection_name", collection_name):metadata の中にある collection_name が、引数で指定したものと一致するデータだけに絞り込んでいます 。
        return result.data or []

    def count_chunks_in_collection(self, collection_name: str) -> int:#select("id", count='exact'):データを取得するのではなく、条件に合うデータの「数」を正確に数えるよう指示しています 。
        result = self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute()
        return result.count or 0#カウント結果を数値で返します 。

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        try:#try: ... except Exception as e::エラーが起きる可能性がある処理を try ブロックに書き、もしエラーが起きたら except ブロックに逃げる（アプリを止めない）構文です 。
            result = self.client.rpc("get_distinct_categories", {"p_collection_name": collection_name}).execute()#result = self.client.rpc(...):get_distinct_categories という関数を呼び出し、カテゴリの一覧を取得します 。
            categories = [item['category'] for item in (result.data or []) if item.get('category')]#取得したデータから「カテゴリ名」だけを取り出してリストに作り変えています（リスト内包表記） 。
            return categories if categories else ["その他"]
        except Exception as e:
            logging.error(f"RPC 'get_distinct_categories' の呼び出しエラー: {e}")#エラーが起きた場合、その内容をログに出力します 。
            return ["その他"]#エラー時やカテゴリが見つからない場合は、安全のため ["その他"] というリストを返します 。

# グローバルインスタンス
db_client: Optional[SupabaseClientManager] = None#このファイルの最後で、db_client という変数を空（None）で定義しています。アプリ起動後に、ここに接続済みの SupabaseClientManager を代入して、他のファイルからも from database import db_client として使えるようにするための場所取りです 。