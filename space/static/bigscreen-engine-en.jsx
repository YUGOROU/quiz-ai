// Quiz Buzzer AI big-screen display — round progression engine (English)
const { useState, useEffect, useRef, useCallback } = React;

// フェーズ: idle → reading → buzzed → answer → judged → (次の問題) ... → roundover
const READ_CPS = 22;          // characters per second (mock STT stream)
const THRESHOLD_BASE = 0.78;  // buzz threshold

// Normalize an answer for forgiving comparison.
// 日本語対応: 英数字に加えてかな/カタカナ/漢字を保持し、句読点・記号・空白を除去。
// ひらがな→カタカナに畳んで表記ゆれを吸収（src/qutils.normalize と同方針の簡易版）。
function normalizeAns(s) {
  let t = (s || "").toLowerCase().normalize("NFKC");
  // 英数字・ひらがな・カタカナ・CJK統合漢字・長音記号以外は空白に
  t = t.replace(/[^a-z0-9ぁ-ゖァ-ヺ一-鿿ー ]/g, " ");
  t = t.replace(/\b(the|a|an)\b/g, " ");
  t = t.replace(/[ぁ-ゖ]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 0x60)); // かな→カナ
  return t.replace(/\s+/g, "").trim();   // 日本語は語間空白が無いので空白は全除去
}
function judgeAnswer(input, truth) {
  const a = normalizeAns(input), b = normalizeAns(truth);
  if (!a || !b) return false;
  return a === b || a.includes(b) || b.includes(a);
}

function BigScreenApp() {
  const round = window.QUIZ_ROUND;
  const total = round.questions.length;

  const [tw, setTw] = useState(() => (window.__twState || {}));
  useEffect(() => {
    const onTw = (s) => setTw({ ...s });
    window.__onTweaks = onTw;
    if (window.__twState) setTw({ ...window.__twState });
    return () => { window.__onTweaks = null; };
  }, []);

  const dir = tw.direction || "competitive";
  const fx = tw.intensity || "flashy";
  const showReasoning = tw.reasoning !== false;
  const autoplay = tw.autoplay !== false;
  const speed = tw.speed || 1;

  const [qIndex, setQIndex] = useState(0);
  const [phase, setPhase] = useState("idle");
  const [seen, setSeen] = useState(0);
  const [confidence, setConfidence] = useState(0.04);
  const [thinkShown, setThinkShown] = useState([]);
  const [flash, setFlash] = useState(null);
  const [revealRest, setRevealRest] = useState(false);
  const [scores, setScores] = useState({
    ai: { pts: 0, correct: 0, wrong: 0, buzzSum: 0, buzzN: 0 },
    human: { pts: 0, correct: 0, wrong: 0, buzzSum: 0, buzzN: 0 },
  });
  const [buzzCompare, setBuzzCompare] = useState({ ai: null, human: null });
  const [lastResult, setLastResult] = useState(null);
  // Live human interaction (English version)
  const [activeBuzzer, setActiveBuzzer] = useState(null); // who actually holds the buzz
  const [humanAnswer, setHumanAnswer] = useState("");      // what the human typed/spoke
  // Rebound（誤答時に未解答の相手へ解答権が移る）
  const [reboundTo, setReboundTo] = useState(null);        // "human" | "ai" | null
  const [aiRebound, setAiRebound] = useState(null);        // {answer, correct} AIが全文で答え返す時
  const [reveal, setReveal] = useState(null);              // {truth} 両者誤答 → 正答を大きく表示

  const timers = useRef([]);
  const clearTimers = () => { timers.current.forEach(clearTimeout); timers.current = []; };
  const at = (ms, fn) => { timers.current.push(setTimeout(fn, ms / speed)); };

  // 問題読み上げ音声（Irodori TTS）。再生位置で buzz fraction を取り、buzz/次問で停止。
  const audioRef = useRef(null);
  const stopAudio = () => { if (audioRef.current) { try { audioRef.current.pause(); } catch (e) {} } };

  // refs mirroring state for use inside event handlers / timers
  const phaseRef = useRef("idle");
  const qIndexRef = useRef(0);
  const seenRef = useRef(0);
  const liveBuzzFrac = useRef(0);
  const runQuestionRef = useRef(null);
  const reboundToRef = useRef(null);
  useEffect(() => { phaseRef.current = phase; }, [phase]);
  useEffect(() => { qIndexRef.current = qIndex; }, [qIndex]);
  useEffect(() => { seenRef.current = seen; }, [seen]);
  useEffect(() => { reboundToRef.current = reboundTo; }, [reboundTo]);

  // 次の問題へ（または終了）
  const advance = (idx) => {
    if (idx + 1 < total) (runQuestionRef.current || runQuestion)(idx + 1);
    else setPhase("roundover");
  };

  const q = round.questions[qIndex];
  const threshold = THRESHOLD_BASE;

  // 1問のタイムラインを構築
  const runQuestion = useCallback((idx) => {
    clearTimers();
    stopAudio();
    const qq = round.questions[idx];
    const len = qq.full.length;
    const buzzChar = Math.round(qq.buzzFrac * len);
    setQIndex(idx);
    setPhase("reading");
    setSeen(0);
    setConfidence(0.04);
    setThinkShown([]);
    setFlash(null);
    setRevealRest(false);
    setLastResult(null);
    setBuzzCompare({ ai: null, human: null });
    setActiveBuzzer(null);
    setHumanAnswer("");
    setReboundTo(null);
    setAiRebound(null);
    setReveal(null);
    // 最初の問題から始めるときはスコアもリセット（リプレイをクリーンに）
    if (idx === 0) {
      setScores({
        ai: { pts: 0, correct: 0, wrong: 0, buzzSum: 0, buzzN: 0 },
        human: { pts: 0, correct: 0, wrong: 0, buzzSum: 0, buzzN: 0 },
      });
    }

    // タイムライン本体。charMs（1文字あたりミリ秒）は音声長 or READ_CPS から決める。
    const buildTimeline = (charMs) => {
      // 文字ストリーム
      for (let c = 1; c <= buzzChar; c++) {
        at(c * charMs, () => setSeen(c));
      }
      // think の段階表示
      (qq.aiThink || []).forEach((t) => {
        const tc = Math.round(t.frac * len);
        at(tc * charMs + 60, () => setThinkShown((prev) => [...prev, t]));
      });
      // confidence ランプ（buzzまで上昇）
      const confSteps = 24;
      for (let s = 1; s <= confSteps; s++) {
        const ms = (buzzChar * charMs) * (s / confSteps);
        const target = qq.buzzer === "ai" ? 0.04 + (0.95 - 0.04) * (s / confSteps)
                                          : 0.04 + (0.62 - 0.04) * (s / confSteps);
        at(ms, () => setConfidence(target));
      }
      // Buzz in（音声もここで停止＝読み上げが buzz 位置で止まる）
      at(buzzChar * charMs + 80, () => {
        stopAudio();
        setSeen(buzzChar);
        setActiveBuzzer(qq.buzzer);
        setPhase("buzzed");
        setBuzzCompare((bc) => ({ ...bc, [qq.buzzer]: qq.buzzFrac }));
        setFlash({ on: true, side: qq.buzzer, key: idx + "-" + Date.now() });
      });
      at(buzzChar * charMs + 80 + 1100, () => setFlash(null));

      if (qq.buzzer === "human") {
        liveBuzzFrac.current = qq.buzzFrac;
        at(buzzChar * charMs + 80 + 1200, () => { setPhase("answering"); setRevealRest(true); });
        return;
      }
      // AI buzzed — auto-resolve (scripted)
      at(buzzChar * charMs + 80 + 1200, () => { setPhase("answer"); setRevealRest(true); });
      at(buzzChar * charMs + 80 + 2600, () => {
        setPhase("judged");
        const reward = 1.0 + 0.5 * (1 - qq.buzzFrac);
        setLastResult({ correct: qq.correct, pts: reward });
        if (window.playResult) window.playResult(qq.correct);   // AI判定の正誤音
        setScores((prev) => {
          const next = JSON.parse(JSON.stringify(prev));
          const me = next.ai;
          me.buzzSum += qq.buzzFrac; me.buzzN += 1;
          if (qq.correct) { me.pts += reward; me.correct += 1; }
          else { me.pts -= 1.5; me.wrong += 1; }
          return next;
        });
        if (qq.correct) {
          if (autoplay) at(3600, () => advance(idx));
        } else {
          at(1600, () => {
            setReboundTo("human"); setActiveBuzzer("human");
            setRevealRest(true); setPhase("answering");
          });
        }
      });
    };

    // 音声があれば長さから charMs を決めて並行再生（開始〜buzzでほぼ同期）。無ければ READ_CPS。
    if (qq.audio) {
      const audio = new Audio(qq.audio);
      audio.preload = "auto";
      audioRef.current = audio;
      let started = false;
      const startWith = (charMs) => { if (started) return; started = true; buildTimeline(charMs); };
      audio.addEventListener("loadedmetadata", () => {
        const dur = (isFinite(audio.duration) && audio.duration > 0.3) ? audio.duration : (len / READ_CPS);
        startWith((dur * 1000) / Math.max(1, len));
        audio.play().catch(() => {});   // 自動再生不可でもタイムラインは進む（無音）
      });
      audio.addEventListener("error", () => startWith(1000 / READ_CPS));
      at(2000, () => startWith(1000 / READ_CPS));  // メタデータ未着の保険
    } else {
      buildTimeline(1000 / READ_CPS);
    }
  }, [round, total, autoplay, speed]);

  // 起動
  useEffect(() => {
    const t = setTimeout(() => runQuestion(0), 700 / speed);
    return () => { clearTimeout(t); clearTimers(); stopAudio(); };
    // eslint-disable-next-line
  }, []);

  // リプレイ（speed/autoplay変更時に手動再生したい場合のフック）
  useEffect(() => {
    window.__replay = () => runQuestion(0);
    runQuestionRef.current = runQuestion;
  }, [runQuestion]);

  // Human buzzes in during reading (click the buzzer or press Space).
  const humanBuzz = useCallback(() => {
    if (phaseRef.current !== "reading") return;
    const idx = qIndexRef.current;
    const qq = round.questions[idx];
    const len = qq.full.length;
    clearTimers();
    // 早押し位置は音声の再生位置を優先（無音時は表示済み文字数）。
    const a = audioRef.current;
    if (a && isFinite(a.duration) && a.duration > 0.3) {
      liveBuzzFrac.current = Math.min(0.98, (a.currentTime || 0) / a.duration);
    } else {
      liveBuzzFrac.current = Math.min(0.98, (seenRef.current || 1) / len);
    }
    stopAudio();
    setActiveBuzzer("human");
    setHumanAnswer("");
    setBuzzCompare((bc) => ({ ...bc, human: liveBuzzFrac.current }));
    setPhase("buzzed");
    setFlash({ on: true, side: "human", key: idx + "-h-" + Date.now() });
    at(1100, () => setFlash(null));
    at(1200, () => { setPhase("answering"); setRevealRest(true); });
  }, [round]);

  // AI が全文で答え返す（人間が誤答した後のリバウンド）。precompute 済 aiFullAnswer を使う。
  const aiReboundResolve = useCallback((idx, qq) => {
    const fa = qq.aiFullAnswer || qq.answer;
    const fc = !!qq.aiFullCorrect;
    setReboundTo("ai");
    setAiRebound({ answer: fa, correct: fc });
    at(2000, () => {
      if (window.playResult) window.playResult(fc);   // AIリバウンド解答の正誤音
      setScores((prev) => {
        const next = JSON.parse(JSON.stringify(prev));
        if (fc) { next.ai.pts += 1.0; next.ai.correct += 1; }
        return next;
      });
      if (!fc) setReveal({ truth: qq.truth });   // 両者誤答 → 正答を大きく表示
      if (autoplay) at(fc ? 3600 : 4200, () => advance(idx));
    });
  }, [autoplay, total]);

  // Human submits a typed / spoken answer.
  const submitHumanAnswer = useCallback((text) => {
    if (phaseRef.current !== "answering") return;
    const idx = qIndexRef.current;
    const qq = round.questions[idx];
    const correct = judgeAnswer(text, qq.truth);
    setHumanAnswer(text);
    setPhase("judged");

    if (reboundToRef.current === "human") {
      // AI 誤答後のリバウンド解答：正解はフラット +1.0、不正解はそれ以上減点しない。
      setLastResult({ correct, pts: 1.0 });
      if (window.playResult) window.playResult(correct);   // 人間リバウンド解答の正誤音
      setScores((prev) => {
        const next = JSON.parse(JSON.stringify(prev));
        if (correct) { next.human.pts += 1.0; next.human.correct += 1; }
        return next;
      });
      if (!correct) setReveal({ truth: qq.truth });  // AI・人間とも誤答 → 正答を大きく表示
      if (autoplay) at(correct ? 3600 : 4200, () => advance(idx));
      return;
    }

    // 通常（人間が早押しした本解答）
    const reward = 1.0 + 0.5 * (1 - liveBuzzFrac.current);
    setLastResult({ correct, pts: reward });
    if (window.playResult) window.playResult(correct);   // 人間の早押し解答の正誤音
    setScores((prev) => {
      const next = JSON.parse(JSON.stringify(prev));
      const me = next.human;
      me.buzzSum += liveBuzzFrac.current; me.buzzN += 1;
      if (correct) { me.pts += reward; me.correct += 1; }
      else { me.pts -= 1.5; me.wrong += 1; }
      return next;
    });
    if (correct) {
      if (autoplay) at(3600, () => advance(idx));
    } else {
      // 人間 誤答 → AI に解答権（全文で答え返す）
      aiReboundResolve(idx, qq);
    }
  }, [round, autoplay, total, aiReboundResolve]);

  // Human passes (no answer).
  const passHuman = useCallback(() => {
    if (phaseRef.current !== "answering") return;
    const idx = qIndexRef.current;
    const qq = round.questions[idx];
    clearTimers();
    if (reboundToRef.current === "human") {
      // AI 誤答後に人間もパス → 両者解答なし → 正答を表示して次へ
      setPhase("judged");
      setReveal({ truth: qq.truth });
      if (autoplay) at(3800, () => advance(idx));
      return;
    }
    advance(idx);
  }, [round, autoplay, total]);

  // Expose handlers + Space-to-buzz.
  useEffect(() => {
    window.__humanBuzz = humanBuzz;
    window.__submitHumanAnswer = submitHumanAnswer;
    window.__passHuman = passHuman;
    const onKey = (e) => {
      if (e.code === "Space" && phaseRef.current === "reading"
        && !/^(INPUT|TEXTAREA)$/.test((e.target && e.target.tagName) || "")) {
        e.preventDefault();
        humanBuzz();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [humanBuzz, submitHumanAnswer, passHuman]);

  const sideUsed = activeBuzzer || (q && q.buzzer);
  const fracUsed = (buzzCompare[sideUsed] != null) ? buzzCompare[sideUsed] : (q && q.buzzFrac);
  const accentLabel = q && phase !== "reading"
    ? (sideUsed === "ai" ? T("ai") : T("human")) + T("buzzAt") + Math.round((fracUsed || 0) * 100) + "%"
    : "";
  const buzzer = (phase === "buzzed" || phase === "answer" || phase === "answering" || phase === "judged") ? activeBuzzer : null;

  return (
    <div className="stage-root" data-dir={dir} data-fx={fx}>
      <TopBar qIndex={qIndex} total={total} phase={phase} />
      <div className="mid">
        <AIColumn q={q} thinkShown={thinkShown}
          phase={phase} buzzer={buzzer} result={buzzer === "ai" ? lastResult : null}
          showReasoning={showReasoning} />
        <CenterBoard scores={scores} />
        <HumanColumn q={q} phase={phase} buzzer={buzzer}
          result={buzzer === "human" ? lastResult : null}
          humanAnswer={humanAnswer}
          onBuzz={humanBuzz} onAnswer={submitHumanAnswer} onPass={passHuman}
          followUp={phase === "judged" && q.followUp ? q.followUp : null} />
      </div>
      <div className="qband">
        <QuestionText q={q} seen={seen} phase={phase} buzzer={buzzer}
          accentLabel={accentLabel} revealRest={revealRest} />
      </div>
      <FlashLayer flash={flash} fx={fx} />
      <ReboundBanner reboundTo={reboundTo} aiRebound={aiRebound} reveal={reveal}
        aiAnswer={q && q.answer} />
      {reveal && <RevealOverlay truth={reveal.truth} />}
      {phase === "roundover" && <RoundOver scores={scores} onReplay={() => runQuestion(0)} />}
    </div>
  );
}

function RoundOver({ scores, onReplay }) {
  const win = scores.ai.correct === scores.human.correct ? "draw" : (scores.ai.correct > scores.human.correct ? "ai" : "human");
  return (
    <div style={{ position: "absolute", inset: 0, zIndex: 50, display: "grid", placeItems: "center",
      background: "color-mix(in oklab, var(--bg) 78%, transparent)", backdropFilter: "blur(6px)" }}>
      <div style={{ textAlign: "center", background: "var(--panel)", border: "1px solid var(--line)",
        borderRadius: 22, padding: "48px 64px", boxShadow: "0 30px 80px rgba(40,38,30,.16)" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 13, letterSpacing: ".1em", textTransform: "uppercase",
          color: "var(--ink2)", marginBottom: 18 }}>{T("result")}</div>
        <div style={{ fontSize: 40, fontWeight: 800, marginBottom: 8,
          color: win === "ai" ? "var(--ai)" : win === "human" ? "var(--hu)" : "var(--ink)" }}>
          {win === "draw" ? T("draw") : win === "ai" ? T("aiWins") : T("humanWins")}
        </div>
        <div style={{ fontSize: 56, fontWeight: 800, fontVariantNumeric: "tabular-nums", marginBottom: 24 }}>
          <span style={{ color: "var(--ai)" }}>{scores.ai.correct}</span>
          <span style={{ color: "var(--ink3)", margin: "0 16px" }}>—</span>
          <span style={{ color: "var(--hu)" }}>{scores.human.correct}</span>
        </div>
        <button onClick={onReplay} style={{ font: "600 15px var(--ui-font)", padding: "12px 28px",
          borderRadius: 999, border: "1px solid var(--line)", background: "var(--ink)", color: "var(--bg)",
          cursor: "pointer" }}>{T("replay")}</button>
      </div>
    </div>
  );
}

// リバウンド（解答権の移動）を上部バナーで知らせる。
function ReboundBanner({ reboundTo, aiRebound, reveal, aiAnswer }) {
  if (!reboundTo || reveal) return null;
  let text = "";
  if (reboundTo === "human") {
    text = T("rebound.aiMissed") + (aiAnswer ? "「" + aiAnswer + "」" : "") + " — " + T("rebound.yourChance");
  } else if (reboundTo === "ai") {
    text = aiRebound
      ? T("rebound.aiTurn") + "「" + aiRebound.answer + "」" + (aiRebound.correct ? " ✓" : " ✕")
      : T("rebound.aiTurn");
  }
  const ok = reboundTo === "ai" && aiRebound && aiRebound.correct;
  return (
    <div style={{ position: "absolute", top: 86, left: "50%", transform: "translateX(-50%)", zIndex: 45,
      background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 999,
      padding: "12px 28px", font: "700 22px var(--ui-font)",
      color: reboundTo === "ai" ? (ok ? "var(--ai)" : "var(--ink)") : "var(--hu)",
      boxShadow: "0 14px 40px rgba(40,38,30,.18)" }}>
      {text}
    </div>
  );
}

// 両者誤答時：正答を大きく表示。
function RevealOverlay({ truth }) {
  return (
    <div style={{ position: "absolute", inset: 0, zIndex: 48, display: "grid", placeItems: "center",
      background: "color-mix(in oklab, var(--bg) 70%, transparent)", backdropFilter: "blur(5px)" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 16, letterSpacing: ".12em",
          color: "var(--ink2)", marginBottom: 14 }}>{T("rebound.correctIs")}</div>
        <div style={{ fontSize: 84, fontWeight: 800, color: "var(--ink)", lineHeight: 1.1 }}>{truth}</div>
      </div>
    </div>
  );
}

window.BigScreenApp = BigScreenApp;
