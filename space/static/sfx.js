// 正誤の効果音を Web Audio API で合成する（音声ファイル非同梱・ライセンス不要・オフライン動作）。
// 正解=明るい2音チャイム（ピンポーン）／不正解=低く濁ったブザー（ブブー）。
// 「明るい高音ベル=正解／濁った低音ブザー=不正解」は日本語話者にも英語話者にも自然に伝わる。
(function () {
  let ctx = null;
  function ac() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
    }
    if (ctx.state === "suspended") { try { ctx.resume(); } catch (e) {} }
    return ctx;
  }

  // 単音を鳴らす（start: 開始までの秒・dur: 長さ秒・type: 波形・gainVal: 音量）。
  function tone(freq, start, dur, type, gainVal) {
    const c = ac();
    if (!c) return;
    const t0 = c.currentTime + start;
    const osc = c.createOscillator();
    const g = c.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    osc.connect(g); g.connect(c.destination);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.linearRampToValueAtTime(gainVal, t0 + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  // 正解: ピンポーン（B5→G5 の2音・ベル風サイン・やや残響感）。
  window.playCorrect = function () {
    if (window.__muteSfx) return;
    try {
      tone(988, 0.00, 0.45, "sine", 0.30);   // ピーン
      tone(784, 0.17, 0.60, "sine", 0.30);   // ポーン
      tone(1318, 0.00, 0.30, "sine", 0.06);  // 倍音で明るさを足す
    } catch (e) {}
  };

  // 不正解: ブブー（低い矩形波の2連・濁ったブザー）。
  window.playWrong = function () {
    if (window.__muteSfx) return;
    try {
      tone(165, 0.00, 0.20, "square", 0.16);  // ブッ
      tone(150, 0.24, 0.42, "square", 0.16);  // ブー
      tone(98, 0.00, 0.50, "sawtooth", 0.05); // 低域でうなりを足す
    } catch (e) {}
  };

  // 判定結果に応じて鳴らすショートカット。
  window.playResult = function (correct) {
    if (correct) window.playCorrect(); else window.playWrong();
  };
})();
