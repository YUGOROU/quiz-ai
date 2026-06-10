// Quiz Buzzer AI — UI 文言の i18n（EN/JA）。
// 問題文は常に日本語（日本特化が差別化軸）。切り替わるのは UI の表示文言のみ。
// プレーン script として babel スクリプトより前にロードする（window.T を先に用意）。
(function () {
  var DICT = {
    ja: {
      // boot
      "boot.title": "Quiz Buzzer AI",
      "boot.sub": "日本語特化スタック（gemma-4-26B-A4B ＋ 自作 buzz タイミングヘッド）で動く、<b>競技早押しクイズ</b>のAI対戦。問題は<span class=\"jp\">日本語</span>。<b>Space</b>キーかタップで早押し。",
      "boot.genreHead": "得意ジャンルを選ぶ",
      "boot.genreAll": "おまかせ（全ジャンル）",
      "boot.start": "対戦スタート",
      "boot.generating": "実モデルで対戦を生成中…（gemma が思考しています）",
      "boot.almost": "まもなく完了…",
      "boot.retry": "もう一度試す",
      "boot.replay": "▶ リプレイ",
      "boot.newMatch": "✦ 新しい対戦",
      "boot.liveLabel": "Quiz Buzzer AI · ライブ",
      // phase
      "phase.idle": "準備完了", "phase.reading": "読み上げ中", "phase.buzzed": "早押し！",
      "phase.answer": "解答", "phase.answering": "解答中", "phase.judged": "判定", "phase.roundover": "終了",
      // answer card / columns
      "q": "第", "qof": "問",
      "ai.answer": "AIの解答", "answer": "解答", "correct": "● 正解", "wrong": "✕ 不正解",
      "buzzerAI": "Buzzer AI", "reasoning": "推論", "reasoningDots": "推論中…",
      "answered": "解答済み", "buzzedIn": "早押し", "standby": "待機中",
      "you": "あなた", "player": "プレイヤー", "listening": "聞き取り中…", "yourTurn": "あなたの番",
      "buzz": "BUZZ", "transcribing": "認識中…",
      "buzzHint.pre": "タップ、または ", "buzzHint.post": " キーで解答",
      // answer input
      "yourAnswer": "あなたの解答", "typePlaceholder": "解答を入力…",
      "micHint": "入力、またはマイクをタップして発声", "typeHint": "解答を入力",
      "pass": "パス", "submit": "解答する",
      "voiceUnsupported": "このブラウザは音声入力に対応していません — 入力してください。",
      "micDenied": "マイクの使用が拒否されました — 入力してください。",
      // center / flash
      "ai": "AI", "human": "人間", "vs": "VS",
      "aiBuzz": "AI 早押し", "humanBuzz": "人間 早押し",
      "buzzAt": " 早押し ‹ ",
      // round over
      "result": "結果", "draw": "引き分け", "aiWins": "AI の勝ち", "humanWins": "あなたの勝ち",
      "replay": "もう一度",
      // rebound（解答権の移動・両者誤答）
      "rebound.aiMissed": "AI が誤答", "rebound.yourChance": "あなたに解答権！",
      "rebound.aiTurn": "AI の解答権 →", "rebound.correctIs": "正解は",
      // 解答制限時間
      "timeLeft": "残り",
      // AI確信度メーター・押下予定
      "confLabel": "AI確信度", "aiPlanned": "AI予定 ‹ ",
      // スコア（ポイント表示）
      "pts": "pt", "correctN": "正解",
      // roundover（振り返り・導線）
      "recap": "振り返り", "watchReplay": "観賞リプレイ",
      "colQ": "問題", "colTruth": "正解", "colAI": "AI", "colYou": "あなた",
      "share": "結果をXでシェア",
      "share.text": "日本語早押しクイズAI「Quiz Buzzer AI」と対戦！結果: 自分 {h} − {a} AI（{hp}pt vs {ap}pt）",
      // エラー（人間語）
      "err.quota": "GPUの利用枠を使い切ったようです。Hugging Face にログインすると枠が増えます。少し時間を置いて、もう一度お試しください。",
      "err.server": "対戦の生成に失敗しました。GPU が混み合っている可能性があります。もう一度お試しください。",
      "err.net": "サーバーに接続できませんでした。通信環境を確認して再試行してください。",
      // 生成待ちのティップス
      "tip.1": "ルール: 問題の読み上げ中、AIより早く BUZZ（Space／タップ）すれば解答権はあなたのもの。",
      "tip.2": "スコア: 正解 +1.0〜1.5（早押しほど高得点）／誤答 −1.5。誤答すると相手に解答権が移ります。",
      "tip.3": "AIの中身: buzz判定は自作1.2B回帰ヘッド、解答は gemma-4-26B-A4B（日本語クイズSFT済）が毎問リアルタイム生成。",
      "tip.4": "ヒント: 左カラムの「AI確信度」メーターがθに迫ったらAIが押す合図。先に押すなら今！",
    },
    en: {
      "boot.title": "Quiz Buzzer AI",
      "boot.sub": "A head-to-head <b>competitive buzz quiz</b> against an AI, built on a Japanese-specialized stack (gemma-4-26B-A4B + a custom buzz-timing head). Questions are in <span class=\"jp\">日本語</span> — buzz in early with <b>Space</b> / tap.",
      "boot.genreHead": "Pick your strong genre",
      "boot.genreAll": "Surprise me (all genres)",
      "boot.start": "Start match",
      "boot.generating": "Generating the match with real models… (gemma is thinking)",
      "boot.almost": "Finishing up…",
      "boot.retry": "Try again",
      "boot.replay": "▶ Replay",
      "boot.newMatch": "✦ New match",
      "boot.liveLabel": "Quiz Buzzer AI · Live",
      "phase.idle": "READY", "phase.reading": "READING", "phase.buzzed": "BUZZ IN",
      "phase.answer": "ANSWER", "phase.answering": "ANSWERING", "phase.judged": "JUDGED", "phase.roundover": "FINAL",
      "q": "Q", "qof": "",
      "ai.answer": "AI’s answer", "answer": "Answer", "correct": "● Correct", "wrong": "✕ Wrong",
      "buzzerAI": "Buzzer AI", "reasoning": "Reasoning", "reasoningDots": "Reasoning…",
      "answered": "Answered", "buzzedIn": "Buzzed in", "standby": "Standby",
      "you": "YOU", "player": "Player", "listening": "Listening…", "yourTurn": "Your turn",
      "buzz": "BUZZ", "transcribing": "Transcribing…",
      "buzzHint.pre": "Tap, or press ", "buzzHint.post": ", to answer",
      "yourAnswer": "Your answer", "typePlaceholder": "Type your answer…",
      "micHint": "Type, or tap the mic to speak", "typeHint": "Type your answer",
      "pass": "Pass", "submit": "Submit",
      "voiceUnsupported": "Voice input isn’t supported in this browser — type instead.",
      "micDenied": "Microphone permission denied — type instead.",
      "ai": "AI", "human": "HUMAN", "vs": "VS",
      "aiBuzz": "AI BUZZ", "humanBuzz": "HUMAN BUZZ",
      "buzzAt": " BUZZ ‹ ",
      "result": "RESULT", "draw": "Draw", "aiWins": "AI wins", "humanWins": "Human wins",
      "replay": "Replay",
      "rebound.aiMissed": "AI missed", "rebound.yourChance": "your chance!",
      "rebound.aiTurn": "AI’s rebound →", "rebound.correctIs": "Correct answer",
      "timeLeft": "Left",
      "confLabel": "AI confidence", "aiPlanned": "AI planned ‹ ",
      "pts": "pt", "correctN": "correct",
      "recap": "Recap", "watchReplay": "Watch replay",
      "colQ": "Question", "colTruth": "Truth", "colAI": "AI", "colYou": "You",
      "share": "Share on X",
      "share.text": "I challenged Quiz Buzzer AI (Japanese buzz-quiz AI)! Result: me {h} − {a} AI ({hp}pt vs {ap}pt)",
      "err.quota": "Looks like the free GPU quota ran out. Signing in to Hugging Face raises your quota — please wait a bit and try again.",
      "err.server": "Failed to generate the match — the GPU may be busy. Please try again.",
      "err.net": "Could not reach the server. Check your connection and retry.",
      "tip.1": "Rule: while the question is being read, buzz (Space / tap) before the AI to claim the answer.",
      "tip.2": "Score: correct +1.0–1.5 (earlier buzz = more points) / wrong −1.5. A miss hands the rebound to your opponent.",
      "tip.3": "Under the hood: buzz timing = our own 1.2B regression head; answers = gemma-4-26B-A4B (Japanese quiz SFT), generated live per question.",
      "tip.4": "Tip: when the AI-confidence meter on the left nears θ, the AI is about to buzz. Beat it!",
    },
  };

  // 既定は日本語（日本特化）。URL ?lang=en / localStorage で上書き。
  var urlLang = new URLSearchParams(location.search).get("lang");
  var saved = (function () { try { return localStorage.getItem("qa_lang"); } catch (e) { return null; } })();
  window.__lang = (urlLang === "en" || urlLang === "ja") ? urlLang : (saved === "en" ? "en" : "ja");

  window.T = function (key) {
    var L = DICT[window.__lang] || DICT.ja;
    return (key in L) ? L[key] : (DICT.ja[key] != null ? DICT.ja[key] : key);
  };
  window.setLang = function (l) {
    window.__lang = (l === "en") ? "en" : "ja";
    try { localStorage.setItem("qa_lang", window.__lang); } catch (e) {}
  };
})();
