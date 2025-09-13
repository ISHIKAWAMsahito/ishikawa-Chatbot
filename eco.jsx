import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, TreePine, Footprints, Clock, Coins, Settings, Trophy, Leaf, Target } from 'lucide-react';

// EcoPomodoroコンポーネント：集中、健康、環境保護を組み合わせたアプリ
const EcoPomodoro = () => {
  // --- 状態管理 (State) ---

  // ポモドーロタイマーの状態
  const [timeLeft, setTimeLeft] = useState(25 * 60); // 残り時間 (秒)
  const [isRunning, setIsRunning] = useState(false);   // タイマー実行中か
  const [isBreak, setIsBreak] = useState(false);       // 休憩時間か
  const [completedPomodoros, setCompletedPomodoros] = useState(0); // 完了したポモドーロ数

  // 歩数計の状態
  const [totalSteps, setTotalSteps] = useState(0);     // 総歩数
  const [todaySteps, setTodaySteps] = useState(0);     // 今日の歩数
  const [stepGoal, setStepGoal] = useState(8000);      // 目標歩数
  const [goalReachedToday, setGoalReachedToday] = useState(false); //目標達成フラグ

  // 寄付・植林の状態
  const [ecoPoints, setEcoPoints] = useState(0);       // エコポイント
  const [treesPlanted, setTreesPlanted] = useState(0); // 植林した木の数
  const [carbonOffset, setCarbonOffset] = useState(0); // CO2削減量 (kg)

  // 設定の状態
  const [workDuration, setWorkDuration] = useState(25); // 作業時間 (分)
  const [breakDuration, setBreakDuration] = useState(5);  // 休憩時間 (分)
  const [activeTab, setActiveTab] = useState('timer');    // アクティブなタブ

  // ---副作用フック (useEffect) ---

  // ポモドーロタイマーのロジック
  useEffect(() => {
    let interval;
    if (isRunning && timeLeft > 0) {
      interval = setInterval(() => {
        setTimeLeft(prevTime => prevTime - 1);
      }, 1000);
    } else if (timeLeft === 0) {
      if (!isBreak) {
        setCompletedPomodoros(prev => prev + 1);
        setEcoPoints(prev => prev + 10);
        setTimeLeft(breakDuration * 60);
        setIsBreak(true);
      } else {
        setTimeLeft(workDuration * 60);
        setIsBreak(false);
      }
      setIsRunning(false);
    }
    return () => clearInterval(interval);
  }, [isRunning, timeLeft, isBreak, workDuration, breakDuration]);

  // 歩数のシミュレーションと目標達成ボーナス
  useEffect(() => {
    const stepInterval = setInterval(() => {
      if (Math.random() > 0.7) {
        const newSteps = Math.floor(Math.random() * 10) + 1;
        const updatedTodaySteps = todaySteps + newSteps;
        
        setTotalSteps(prev => prev + newSteps);
        setTodaySteps(updatedTodaySteps);
        
        // 100歩ごとにポイント獲得
        if (Math.floor(updatedTodaySteps / 100) > Math.floor(todaySteps / 100)) {
            setEcoPoints(prev => prev + 1);
        }

        // 目標達成時に一度だけボーナスポイントを付与
        if (updatedTodaySteps >= stepGoal && !goalReachedToday) {
          setEcoPoints(prev => prev + 20); // ボーナスポイント
          setGoalReachedToday(true);
        }
      }
    }, 2000);
    return () => clearInterval(stepInterval);
  }, [todaySteps, stepGoal, goalReachedToday]);

  // --- ヘルパー関数 ---
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // --- イベントハンドラ ---
  const toggleTimer = () => setIsRunning(!isRunning);

  const resetTimer = () => {
    setIsRunning(false);
    setIsBreak(false);
    setTimeLeft(workDuration * 60);
  };

  const plantTree = () => {
    if (ecoPoints >= 50) {
      setEcoPoints(prev => prev - 50);
      setTreesPlanted(prev => prev + 1);
      setCarbonOffset(prev => prev + 22);
    }
  };

  const donateFunds = () => {
    if (ecoPoints >= 100) {
      setEcoPoints(prev => prev - 100);
      setCarbonOffset(prev => prev + 50);
    }
  };
  
  const stepProgress = Math.min((todaySteps / stepGoal) * 100, 100);

  // --- レンダリング (JSX) ---
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4 font-sans">
      <div className="max-w-md mx-auto">
        <header className="text-center mb-6">
          <h1 className="text-3xl font-bold text-green-800 flex items-center justify-center gap-2">
            <TreePine className="text-green-600" />
            エコ・フォーカス
          </h1>
          <p className="text-green-600 text-sm">集中 × 健康 × 環境保護</p>
        </header>

        <nav className="flex bg-white rounded-lg p-1 mb-6 shadow-sm">
          {[
            { id: 'timer', icon: Clock, label: 'タイマー' },
            { id: 'steps', icon: Footprints, label: '歩数' },
            { id: 'forest', icon: TreePine, label: '森林' },
            { id: 'settings', icon: Settings, label: '設定' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-lg transition-all ${
                activeTab === tab.id 
                  ? 'bg-green-500 text-white shadow-md' 
                  : 'text-gray-600 hover:bg-green-50'
              }`}
            >
              <tab.icon size={20} />
              <span className="text-sm font-medium">{tab.label}</span>
            </button>
          ))}
        </nav>

        <section className="bg-white rounded-lg p-4 mb-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Coins className="text-yellow-500" size={20} />
              <span className="font-medium text-gray-700">エコポイント</span>
            </div>
            <span className="text-2xl font-bold text-green-600">{ecoPoints}</span>
          </div>
        </section>

        <main>
          {activeTab === 'timer' && (
            <div className="space-y-6">
              <section className="bg-white rounded-lg p-6 shadow-sm text-center">
                <div className={`text-6xl font-mono font-bold mb-4 ${isBreak ? 'text-blue-600' : 'text-green-600'}`}>
                  {formatTime(timeLeft)}
                </div>
                <p className="text-gray-600 mb-6">
                  {isBreak ? '休憩中 🌱' : '集中タイム 🎯'}
                </p>
                <div className="flex gap-3 justify-center">
                  <button onClick={toggleTimer} className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${ isRunning ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'}`}>
                    {isRunning ? <Pause size={20} /> : <Play size={20} />}
                    {isRunning ? '一時停止' : '開始'}
                  </button>
                  <button onClick={resetTimer} className="flex items-center gap-2 px-6 py-3 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-all">
                    <Square size={20} />
                    リセット
                  </button>
                </div>
              </section>
              <section className="bg-white rounded-lg p-4 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
                  <Trophy className="text-yellow-500" size={20} />今日の実績
                </h3>
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-green-600">{completedPomodoros}</div>
                    <div className="text-sm text-gray-600">完了セッション</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-blue-600">
                      {Math.floor(completedPomodoros * workDuration / 60)}h {(completedPomodoros * workDuration) % 60}m
                    </div>
                    <div className="text-sm text-gray-600">集中時間</div>
                  </div>
                </div>
              </section>
            </div>
          )}

          {activeTab === 'steps' && (
             <div className="space-y-6">
              <section className="bg-white rounded-lg p-6 shadow-sm text-center">
                <div className="text-5xl font-bold text-blue-600 mb-2">{todaySteps.toLocaleString()}</div>
                <p className="text-gray-600 mb-4">今日の歩数</p>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                  <div className="bg-blue-500 h-3 rounded-full transition-all duration-500" style={{ width: `${stepProgress}%` }}></div>
                </div>
                <p className="text-sm text-gray-600">目標: {stepGoal.toLocaleString()}歩 ({Math.round(stepProgress)}%)</p>
              </section>
              <section className="bg-white rounded-lg p-4 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
                  <Footprints className="text-blue-500" size={20} />歩数データ
                </h3>
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-blue-600">{totalSteps.toLocaleString()}</div>
                    <div className="text-sm text-gray-600">総歩数</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-orange-600">{ (todaySteps * 0.7 / 1000).toFixed(2) }km</div>
                    <div className="text-sm text-gray-600">今日の距離</div>
                  </div>
                </div>
              </section>
              {goalReachedToday && (
                <div className="mt-4 bg-green-100 border-l-4 border-green-500 p-4 rounded">
                  <p className="text-green-800 font-medium">🎉 歩数目標達成！ボーナス20ポイント獲得！</p>
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'forest' && (
            <div className="space-y-6">
              <section className="bg-white rounded-lg p-6 shadow-sm text-center">
                <div className="text-4xl mb-2">🌳</div>
                <div className="text-3xl font-bold text-green-600 mb-2">{treesPlanted}</div>
                <p className="text-gray-600 mb-4">植林した木</p>
                <div className="text-sm text-gray-600">CO2削減量: <span className="font-bold text-green-600">{carbonOffset}kg</span></div>
              </section>
              <section className="space-y-4">
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <TreePine className="text-green-500" size={24} />
                      <div>
                        <h3 className="font-bold text-gray-800">木を植える</h3>
                        <p className="text-sm text-gray-600">50ポイントで1本植林</p>
                      </div>
                    </div>
                    <button onClick={plantTree} disabled={ecoPoints < 50} className={`px-4 py-2 rounded-lg font-medium transition-all ${ ecoPoints >= 50 ? 'bg-green-500 hover:bg-green-600 text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}>植林</button>
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <Leaf className="text-blue-500" size={24} />
                      <div>
                        <h3 className="font-bold text-gray-800">環境団体に寄付</h3>
                        <p className="text-sm text-gray-600">100ポイントで寄付</p>
                      </div>
                    </div>
                    <button onClick={donateFunds} disabled={ecoPoints < 100} className={`px-4 py-2 rounded-lg font-medium transition-all ${ ecoPoints >= 100 ? 'bg-blue-500 hover:bg-blue-600 text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}>寄付</button>
                  </div>
                </div>
              </section>
              <section className="bg-white rounded-lg p-4 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-3">ポイント獲得方法</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between items-center"><span className="flex items-center gap-2"><Clock size={16} className="text-green-500" />ポモドーロ完了</span><span className="font-bold text-green-600">+10pt</span></li>
                  <li className="flex justify-between items-center"><span className="flex items-center gap-2"><Footprints size={16} className="text-blue-500" />100歩ごと</span><span className="font-bold text-blue-600">+1pt</span></li>
                  <li className="flex justify-between items-center"><span className="flex items-center gap-2"><Trophy size={16} className="text-yellow-500" />歩数目標達成</span><span className="font-bold text-yellow-600">+20pt</span></li>
                </ul>
              </section>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="space-y-6">
              <section className="bg-white rounded-lg p-4 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2"><Settings size={20} />設定</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">作業時間 (分)</label>
                    <select value={workDuration} onChange={(e) => { setWorkDuration(Number(e.target.value)); if (!isRunning && !isBreak) setTimeLeft(Number(e.target.value) * 60); }} className="w-full p-2 border border-gray-300 rounded-lg text-sm">
                      <option value={15}>15</option><option value={25}>25</option><option value={45}>45</option><option value={60}>60</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">休憩時間 (分)</label>
                    <select value={breakDuration} onChange={(e) => setBreakDuration(Number(e.target.value))} className="w-full p-2 border border-gray-300 rounded-lg text-sm">
                      <option value={5}>5</option><option value={10}>10</option><option value={15}>15</option><option value={20}>20</option>
                    </select>
                  </div>
                   <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">歩数目標</label>
                    <select value={stepGoal} onChange={(e) => setStepGoal(Number(e.target.value))} className="w-full p-2 border border-gray-300 rounded-lg text-sm">
                      <option value={5000}>5,000</option><option value={8000}>8,000</option><option value={10000}>10,000</option><option value={12000}>12,000</option>
                    </select>
                  </div>
                </div>
              </section>
            </div>
          )}
        </main>

        <footer className="mt-8 text-center">
          <p className="text-xs text-gray-500">
            集中して、歩いて、地球を守ろう 🌍
          </p>
        </footer>
      </div>
    </div>
  );
};

export default EcoPomodoro;
