# 1. シナリオの定義
SCENARIOS = {
    'leave_of_absence': {
        'trigger_keywords': ['休学したい', '休学手続き'],
        'steps': {
            0: "休学をご希望ですね。手続きをご案内します。理由を簡単に入力してください。（例：病気のため、留学のため）",
            1: "承知いたしました。「{user_input}」が理由ですね。次のステップとして、学生支援課で正式な手続きが必要です。必要な書類など、詳細は学生支援課にご確認ください。これで手続き案内を終了します。"
        }
    }
}

# 2. シナリオ処理を行う関数
def process_scenario_chat(query, session_state):
    """
    シナリオ処理をシミュレートする関数
    - query: ユーザーからの入力文字列
    - session_state: 現在のセッション状態を模した辞書
    - 戻り値: (応答メッセージ, 更新されたセッション状態) のタプル
    """
    updated_state = session_state.copy()
    scenario_state = updated_state.get('scenario_state')

    # --- 1. 進行中のシナリオがあれば処理 ---
    if scenario_state:
        name = scenario_state.get('name')
        step = scenario_state.get('step')
        scenario = SCENARIOS.get(name)

        if scenario and (step + 1) in scenario['steps']:
            # 次のステップのメッセージを返し、シナリオを終了（状態をクリア）
            response = scenario['steps'][step + 1].format(user_input=query)
            updated_state.pop('scenario_state', None)
            return response, updated_state

    # --- 2. 新しいシナリオを開始するか判定 ---
    for name, scenario in SCENARIOS.items():
        if any(keyword in query for keyword in scenario['trigger_keywords']):
            # シナリオを開始し、ステップ0の状態をセッションに保存
            updated_state['scenario_state'] = {'name': name, 'step': 0}
            response = scenario['steps'][0]
            return response, updated_state

    # --- 3. シナリオに該当しない場合 ---
    return f"（シナリオ対象外: '{query}'） -> 通常のFAQ応答へ", updated_state


# 3. ターミナルで対話をテストするための実行ブロック
if __name__ == "__main__":
    print("シナリオチャットテストを開始します。（'exit'で終了）")
    session = {}
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'exit':
            break
        
        bot_response, session = process_scenario_chat(user_input, session)
        print(f"ボット: {bot_response}")
        print(f"  (現在のセッション: {session})\n")
