import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import google.generativeai as genai

from core.dependencies import require_auth
from core import database
from core import settings
# 【追加】APIキーを確実に読み込む
from core.config import GEMINI_API_KEY

router = APIRouter()

# 【追加】モジュール読み込み時にAPIキーを設定する
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@router.get("/api/fallbacks")
async def get_all_fallbacks(user: dict = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        # "embedding" カラムも select する
        result = database.db_client.client.table("category_fallbacks").select("id, question, answer, category_name, embedding").order("id", desc=True).execute()
        return {"fallbacks": result.data or []}
    except Exception as e:
        logging.error(f"Q&A一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/fallbacks")
async def create_fallback(request: Dict[str, Any], user: dict = Depends(require_auth)):
    """新しいQ&Aを作成(保存時に「質問」を自動でベクトル化)"""
    if not database.db_client or not settings.settings_manager: 
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    
    try:
        question_text = request.get("question")
        answer_text = request.get("answer")
        category_name = request.get("category_name")
        
        if not question_text or not answer_text or not category_name:
             raise HTTPException(status_code=400, detail="question, answer, category_name は必須です")

        embedding = None
        try:
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"新規Q&Aの「質問」のベクトルを生成します...")
            
            # question_text (質問文) をベクトル化
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=question_text
            )
            embedding = embedding_response["embedding"]
            logging.info(f"新規Q&Aの「質問」のベクトル生成が完了しました。")
        
        except Exception as e:
            logging.error(f"新規Q&Aのベクトル生成エラー: {e}")
            logging.warning(f"ベクトル化に失敗しましたが、テキストは保存します。")

        insert_data = {
            "question": question_text,
            "answer": answer_text,
            "category_name": category_name,
            "embedding": embedding
        }
        
        result = database.db_client.client.table("category_fallbacks").insert(insert_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Q&Aの作成に失敗しました")

        logging.info(f"新規Q&A {result.data[0]['id']} を作成しました(管理者: {user.get('email')})")
        
        message = "新しいQ&Aを保存し、ベクトル化も完了しました。" if embedding else "新しいQ&Aを保存しました(ベクトル化には失敗)"
        return {"message": message, "fallback": result.data[0]}
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Q&A作成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/fallbacks/{qa_id}")
async def update_fallback(qa_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """Q&Aを更新(「質問」変更時に自動でベクトル化)"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}
        
        if "question" in request:
            new_question = request["question"]
            if not new_question or not new_question.strip():
                raise HTTPException(status_code=400, detail="question を空にすることはできません")
            
            update_data["question"] = new_question
            
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"Q&A {qa_id} の「質問」が変更されたため、ベクトルを再生成します...")
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_question
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"Q&A {qa_id} のベクトル再生成が完了しました。")
            except Exception as e:
                logging.error(f"Q&Aベクトル再生成エラー: {e}")
                update_data["embedding"] = None
                logging.warning(f"Q&A {qa_id} のベクトル化に失敗しましたが、テキストは更新します。")

        if "answer" in request:
            new_answer = request["answer"]
            if not new_answer or not new_answer.strip():
                 raise HTTPException(status_code=400, detail="answer を空にすることはできません")
            update_data["answer"] = new_answer
            
        if "category_name" in request:
            new_category = request.get("category_name")
            if not new_category or not new_category.strip():
                 raise HTTPException(status_code=400, detail="category_name を空にすることはできません")
            update_data["category_name"] = new_category

        if not update_data:
            raise HTTPException(status_code=400, detail="更新するデータがありません")
        
        result = database.db_client.client.table("category_fallbacks").update(update_data).eq("id", qa_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Q&Aが見つかりません")
        
        logging.info(f"Q&A {qa_id} を更新しました(管理者: {user.get('email')})")
        return {"message": "Q&Aを更新しました", "fallback": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Q&A更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/fallbacks/{qa_id}")
async def delete_fallback(qa_id: int, user: dict = Depends(require_auth)):
    """Q&Aを削除"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("category_fallbacks").delete().eq("id", qa_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Q&Aが見つかりません")
        
        logging.info(f"Q&A {qa_id} を削除しました(管理者: {user.get('email')})")
        return {"message": "Q&Aを削除しました"}
    except Exception as e:
        logging.error(f"Q&A削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ▼▼▼▼▼▼▼▼ 【診断テスト用に修正した関数】 ▼▼▼▼▼▼▼▼
@router.post("/api/fallbacks/vectorize-all")
async def vectorize_all_missing_fallbacks(user: dict = Depends(require_auth)):
    """全Q&Aをテスト文字列で強制的に再ベクトル化する(診断用)"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")

    logging.info(f"【診断モード】全Q&Aの強制ベクトル化テストを開始...(管理者: {user.get('email')})")
    
    try:
        # ★修正: embeddingがNULLのものだけでなく、すべてのデータを取得する
        response = database.db_client.client.table("category_fallbacks").select("id, question").execute()
        
        if not response.data:
            return {"message": "処理対象のQ&Aがありませんでした。"}

        embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
        count = 0
        
        for item in response.data:
            item_id = item['id']
            
            # ★修正: データベースの質問文を無視し、固定のテスト文字列を使用する
            # これにより、文字コード等の問題を排除してAPIの疎通確認を行う
            text_to_vectorize = f"TEST_VECTOR_FOR_ID_{item_id}"

            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=text_to_vectorize
                )
                new_embedding = embedding_response["embedding"]

                # DBを更新
                database.db_client.client.table("category_fallbacks").update({
                    "embedding": new_embedding
                }).eq("id", item_id).execute()
                
                logging.info(f"Q&A ID {item_id}: テスト文字列 '{text_to_vectorize}' でベクトル化完了。")
                count += 1
                await asyncio.sleep(1) # APIレート制限対策

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning(f"APIレート制限のため30秒待機します... (ID {item_id})")
                    await asyncio.sleep(30)
                    # 再試行
                    embedding_response = genai.embed_content(model=embedding_model, content=text_to_vectorize)
                    new_embedding = embedding_response["embedding"]
                    database.db_client.client.table("category_fallbacks").update({"embedding": new_embedding}).eq("id", item_id).execute()
                    logging.info(f"Q&A ID {item_id}: (再試行) ベクトル化完了。")
                    count += 1
                else:
                    logging.error(f"Q&A ID {item_id} のベクトル化エラー: {e}")

        logging.info(f"全Q&A診断処理完了。 {count}件を更新しました。")
        return {"message": f"【診断完了】全{count}件をテスト用ベクトルで上書きしました。DBを確認してください。"}

    except Exception as e:
        logging.error(f"診断処理中にエラーが発生: {e}")
        raise HTTPException(status_code=500, detail=str(e))