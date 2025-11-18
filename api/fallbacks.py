import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import google.generativeai as genai
from core.config import GEMINI_API_KEY
from core.dependencies import require_auth
from core import database
from core import settings

router = APIRouter()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
@router.get("/api/fallbacks")
async def get_all_fallbacks(user: dict = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        # ▼▼▼ [ここから修正] ▼▼▼
        # "embedding" カラムも select するように追加
        result = database.db_client.client.table("category_fallbacks").select("id, question, answer, category_name, embedding").order("id", desc=True).execute()
        # ▲▲▲ [ここまで修正] ▲▲▲
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
        # ★修正: static_response ではなく question と answer を取得
        question_text = request.get("question")
        answer_text = request.get("answer")
        category_name = request.get("category_name")
        
        if not question_text or not answer_text or not category_name:
             raise HTTPException(status_code=400, detail="question, answer, category_name は必須です")

        embedding = None
        try:
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"新規Q&Aの「質問」のベクトルを生成します...")
            
            # ★修正: question_text (質問文) だけをベクトル化
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=question_text
            )
            embedding = embedding_response["embedding"]
            logging.info(f"新規Q&Aの「質問」のベクトル生成が完了しました。")
        
        except Exception as e:
            logging.error(f"新規Q&Aのベクトル生成エラー: {e}")
            logging.warning(f"ベクトル化に失敗しましたが、テキストは保存します。")

        # ★修正: question と answer を個別に保存
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
        
        # ★修正: 「質問」が変更された場合
        if "question" in request:
            new_question = request["question"]
            if not new_question or not new_question.strip():
                raise HTTPException(status_code=400, detail="question を空にすることはできません")
            
            update_data["question"] = new_question
            
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"Q&A {qa_id} の「質問」が変更されたため、ベクトルを再生成します...")
            try:
                # ★修正: new_question (質問文) だけをベクトル化
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

        # ★修正: 「回答」が変更された場合 (ベクトル化は不要)
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
    # (この関数は変更不要)
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

@router.post("/api/fallbacks/vectorize-all")
async def vectorize_all_missing_fallbacks(user: dict = Depends(require_auth)):
    """embedding が NULL のQ&Aをすべてベクトル化する"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")

    logging.info(f"全Q&Aのベクトル化処理を開始...(管理者: {user.get('email')})")
    
    try:
        # ★修正: static_response ではなく question を取得
        response = database.db_client.client.table("category_fallbacks").select("id, question").is_("embedding", "null").execute()
        
        if not response.data:
            return {"message": "ベクトル化が必要なQ&Aはありませんでした。"}

        embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
        count = 0
        
        for item in response.data:
            item_id = item['id']
            # ★修正: static_response ではなく question をベクトル化
            text_to_vectorize = item['question']

            if not text_to_vectorize or not text_to_vectorize.strip():
                logging.warning(f"Q&A ID {item_id}: 質問(question)が空のためスキップします。")
                continue

            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=text_to_vectorize
                )
                new_embedding = embedding_response["embedding"]

                database.db_client.client.table("category_fallbacks").update({
                    "embedding": new_embedding
                }).eq("id", item_id).execute()
                
                logging.info(f"Q&A ID {item_id}: ベクトル化完了。")
                count += 1
                await asyncio.sleep(1) # APIレート制限を避けるため、1秒待機

            except Exception as e:
                # (レート制限のロジックは変更なし)
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

        logging.info(f"全Q&Aベクトル化処理完了。 {count}件を処理しました。")
        return {"message": f"ベクトル化処理が完了しました。{count}件のQ&Aを更新しました。"}

    except Exception as e:
        logging.error(f"全Q&Aベクトル化処理中にエラーが発生: {e}")
        raise HTTPException(status_code=500, detail=str(e))