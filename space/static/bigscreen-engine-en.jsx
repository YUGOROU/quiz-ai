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
// kana: 正解の読み仮名（truth_kana・pool 生成時に pykakasi で付与）。
// 漢字の正解に「なつめそうせき」等のかな解答を救済するため、truth と両方に照合する。
function judgeAnswer(input, truth, kana) {
  const a = normalizeAns(input);
  if (!a) return false;
  return [truth, kana].filter(Boolean).some((t) => {
    const b = normalizeAns(t);
    return b && (a === b || a.includes(b) || b.includes(a));
  });
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
  // 解答制限時間（buzz 後に無制限に考えられると AI に不公平・テンポも死ぬ）
  const ANSWER_SEC = 12;
  const [answerLeft, setAnswerLeft] = useState(null);
  // 振り返り（roundover の全問サマリー用・問題 idx → 両者の解答記録）
  const outcomesRef = useRef({});
  const recordOutcome = (idx, patch) => {
    outcomesRef.current[idx] = Object.assign({}, outcomesRef.current[idx], patch);
  };

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
    // 最初の問題から始めるときはスコアと振り返りもリセット（リプレイをクリーンに）
    if (idx === 0) {
      setScores({
        ai: { pts: 0, correct: 0, wrong: 0, buzzSum: 0, buzzN: 0 },
        human: { pts: 0, correct: 0, wrong: 0, buzzSum: 0, buzzN: 0 },
      });
      outcomesRef.current = {};
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
      // confidence ランプ。実測カーブ（buzz回帰ヘッドの confCurve）があれば本物を再生、
      // 無ければ線形フォールバック（mock 等）。
      if (qq.confCurve && qq.confCurve.length) {
        qq.confCurve.forEach((pt) => {
          const tc = Math.round(pt.f * len);
          if (tc <= buzzChar) at(tc * charMs, () => setConfidence(pt.c));
        });
      } else {
        const confSteps = 24;
        for (let s = 1; s <= confSteps; s++) {
          const ms = (buzzChar * charMs) * (s / confSteps);
          const target = qq.buzzer === "ai" ? 0.04 + (0.95 - 0.04) * (s / confSteps)
                                            : 0.04 + (0.62 - 0.04) * (s / confSteps);
          at(ms, () => setConfidence(target));
        }
      }
      // Buzz in（音声もここで停止＝読み上げが buzz 位置で止まる）
      at(buzzChar * charMs + 80, () => {
        stopAudio();
        setSeen(buzzChar);
        setActiveBuzzer(qq.buzzer);
        setPhase("buzzed");
        setBuzzCompare((bc) => ({ ...bc, [qq.buzzer]: qq.buzzFrac }));
        setFlash({ on: true, side: qq.buzzer, key: idx + "-" + Date.now() });
        if (window.playBuzz) window.playBuzz(qq.buzzer);
      });
      at(buzzChar * charMs + 80 + 1100, () => setFlash(null));

      if (qq.buzzer === "human") {
        liveBuzzFrac.current = qq.buzzFrac;
        // 解答が確定するまで問題文の続きは見せない（カンニング防止）。
        at(buzzChar * charMs + 80 + 1200, () => setPhase("answering"));
        return;
      }
      // AI buzzed — auto-resolve (scripted)
      at(buzzChar * charMs + 80 + 1200, () => { setPhase("answer"); setRevealRest(true); });
      at(buzzChar * charMs + 80 + 2600, () => {
        setPhase("judged");
        const reward = 1.0 + 0.5 * (1 - qq.buzzFrac);
        setLastResult({ correct: qq.correct, pts: reward,
          delta: qq.correct ? reward : -1.5 });
        if (window.playResult) window.playResult(qq.correct);   // AI判定の正誤音
        recordOutcome(idx, { aiAns: qq.answer, aiOk: qq.correct, aiFrac: qq.buzzFrac });
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
      audio.muted = !!window.__muteAll;   // nav のミュートトグルに追従
      audio.playbackRate = speed;         // タイムラインは at() が /speed するので音声側も同期
      window.__curAudio = audio;
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
    if (window.playBuzz) window.playBuzz("human");
    at(1100, () => setFlash(null));
    // 解答確定まで問題文の続きは隠す（早押しの意味を守る・カンニング防止）。
    at(600, () => setPhase("answering"));
  }, [round]);

  // AI が全文で答え返す（人間が誤答／パスした後のリバウンド）。precompute 済 aiFullAnswer を使う。
  const aiReboundResolve = useCallback((idx, qq) => {
    const fa = qq.aiFullAnswer || qq.answer;
    const fc = !!qq.aiFullCorrect;
    setReboundTo("ai");
    setAiRebound({ answer: fa, correct: fc });
    setRevealRest(true);   // AIは全文で答えるので、観客にも全文を見せる
    at(2000, () => {
      if (window.playResult) window.playResult(fc);   // AIリバウンド解答の正誤音
      recordOutcome(idx, { aiAns: fa, aiOk: fc, aiRebound: true });
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
    const correct = judgeAnswer(text, qq.truth, qq.truthKana);
    setHumanAnswer(text);
    setPhase("judged");
    setRevealRest(true);   // 解答が確定したのでここで全文を開示

    if (reboundToRef.current === "human") {
      // AI 誤答後のリバウンド解答：正解はフラット +1.0、不正解はそれ以上減点しない。
      setLastResult({ correct, pts: 1.0, delta: correct ? 1.0 : 0 });
      if (window.playResult) window.playResult(correct);   // 人間リバウンド解答の正誤音
      recordOutcome(idx, { humanAns: text, humanOk: correct, humanRebound: true });
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
    setLastResult({ correct, pts: reward, delta: correct ? reward : -1.5 });
    if (window.playResult) window.playResult(correct);   // 人間の早押し解答の正誤音
    recordOutcome(idx, { humanAns: text, humanOk: correct, humanFrac: liveBuzzFrac.current });
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

  // Human passes (no answer / time up).
  const passHuman = useCallback(() => {
    if (phaseRef.current !== "answering") return;
    const idx = qIndexRef.current;
    const qq = round.questions[idx];
    clearTimers();
    setPhase("judged");
    if (reboundToRef.current === "human") {
      // AI 誤答後に人間もパス → 両者解答なし → 正答を表示して次へ
      setRevealRest(true);
      setReveal({ truth: qq.truth });
      if (autoplay) at(3800, () => advance(idx));
      return;
    }
    // 人間が早押し後にパス → 誤答と同じく AI に解答権が移る（ルール一貫・正答も必ず見える）。
    recordOutcome(idx, { humanAns: "", humanOk: false, humanFrac: liveBuzzFrac.current });
    aiReboundResolve(idx, qq);
  }, [round, autoplay, total, aiReboundResolve]);

  // 解答制限時間: answering になったらカウントダウン、0 でパス扱い。
  const passRef = useRef(null);
  useEffect(() => { passRef.current = passHuman; }, [passHuman]);
  useEffect(() => {
    if (phase !== "answering") { setAnswerLeft(null); return; }
    setAnswerLeft(ANSWER_SEC);
    const iv = setInterval(() => {
      setAnswerLeft((p) => {
        if (p == null) return p;
        if (p <= 1) {
          clearInterval(iv);
          if (passRef.current) passRef.current();
          return 0;
        }
        return p - 1;
      });
    }, 1000);
    return () => clearInterval(iv);
  }, [phase, qIndex]);

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
  const theta = round.theta || THRESHOLD_BASE;
  // 人間が AI より先に押した問題は、判定後に「AI はここで押す予定だった」を問題文中に示す。
  const aiPlanFrac = (activeBuzzer === "human" && phase === "judged" && q) ? q.buzzFrac : null;

  return (
    <div className="stage-root" data-dir={dir} data-fx={fx}>
      <TopBar qIndex={qIndex} total={total} phase={phase} genre={q && q.genre} />
      <div className="mid">
        <AIColumn q={q} thinkShown={thinkShown}
          phase={phase} buzzer={buzzer} result={buzzer === "ai" ? lastResult : null}
          showReasoning={showReasoning} confidence={confidence} theta={theta} />
        <CenterBoard scores={scores} />
        <HumanColumn q={q} phase={phase} buzzer={buzzer}
          result={buzzer === "human" ? lastResult : null}
          humanAnswer={humanAnswer} timeLeft={answerLeft}
          onBuzz={humanBuzz} onAnswer={submitHumanAnswer} onPass={passHuman}
          followUp={phase === "judged" && q.followUp ? q.followUp : null} />
      </div>
      <div className="qband">
        <QuestionText q={q} seen={seen} phase={phase} buzzer={buzzer}
          accentLabel={accentLabel} revealRest={revealRest} aiPlan={aiPlanFrac} />
      </div>
      <FlashLayer flash={flash} fx={fx} />
      <ReboundBanner reboundTo={reboundTo} aiRebound={aiRebound} reveal={reveal}
        aiAnswer={q && q.answer} />
      {reveal && <RevealOverlay truth={reveal.truth} />}
      {phase === "roundover" && (
        <RoundOver scores={scores} round={round} outcomes={outcomesRef.current}
          onReplay={() => runQuestion(0)}
          onNew={() => { if (window.__newMatch) window.__newMatch(); }} />
      )}
    </div>
  );
}

function RoundOver({ scores, round, outcomes, onReplay, onNew }) {
  // 勝敗はポイント（早押しボーナス・誤答ペナルティ込み）で決める。正解数は併記。
  const d = scores.ai.pts - scores.human.pts;
  const win = Math.abs(d) < 0.001 ? "draw" : (d > 0 ? "ai" : "human");
  const fmt = (v) => (Math.round(v * 10) / 10).toFixed(1);

  const share = () => {
    const text = T("share.text")
      .replace("{h}", scores.human.correct).replace("{a}", scores.ai.correct)
      .replace("{hp}", fmt(scores.human.pts)).replace("{ap}", fmt(scores.ai.pts));
    const url = "https://huggingface.co/spaces/build-small-hackathon/quiz-buzzer-ai";
    window.open("https://twitter.com/intent/tweet?text=" + encodeURIComponent(text)
      + "&url=" + encodeURIComponent(url), "_blank", "noopener");
  };

  const mark = (ok) => (ok ? <span className="rc-ok">●</span> : <span className="rc-ng">✕</span>);
  const qs = (round && round.questions) || [];

  return (
    <div className="roundover-veil">
      <div className="roundover-card">
        <div className="ro-label">{T("result")}</div>
        <div className="ro-winner" style={{
          color: win === "ai" ? "var(--ai)" : win === "human" ? "var(--hu)" : "var(--ink)" }}>
          {win === "draw" ? T("draw") : win === "ai" ? T("aiWins") : T("humanWins")}
        </div>
        <div className="ro-pts">
          <span style={{ color: "var(--ai)" }}>{fmt(scores.ai.pts)}<small>{T("pts")}</small></span>
          <span className="ro-dash">—</span>
          <span style={{ color: "var(--hu)" }}>{fmt(scores.human.pts)}<small>{T("pts")}</small></span>
        </div>
        <div className="ro-sub">
          {T("correctN")}: <b style={{ color: "var(--ai)" }}>{scores.ai.correct}</b>
          <span className="ro-dim"> vs </span>
          <b style={{ color: "var(--hu)" }}>{scores.human.correct}</b>
        </div>
        {qs.length > 0 && (
          <div className="recap">
            <div className="recap-head">{T("recap")}</div>
            <table className="recap-table">
              <thead>
                <tr><th></th><th className="rc-q">{T("colQ")}</th><th>{T("colTruth")}</th>
                  <th>{T("colAI")}</th><th>{T("colYou")}</th></tr>
              </thead>
              <tbody>
                {qs.map((qq, i) => {
                  const o = (outcomes && outcomes[i]) || {};
                  return (
                    <tr key={i}>
                      <td className="rc-no">{i + 1}</td>
                      <td className="rc-q">{qq.full.length > 26 ? qq.full.slice(0, 26) + "…" : qq.full}</td>
                      <td className="rc-truth">{qq.truth}</td>
                      <td>{o.aiAns != null ? <span>{o.aiAns || "—"} {mark(o.aiOk)}</span> : <span className="ro-dim">—</span>}</td>
                      <td>{o.humanAns != null ? <span>{o.humanAns || T("pass")} {mark(o.humanOk)}</span> : <span className="ro-dim">—</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <div className="ro-actions">
          <button className="ro-btn primary" onClick={onNew}>{T("boot.newMatch")}</button>
          <button className="ro-btn" onClick={onReplay}>{T("watchReplay")}</button>
          <button className="ro-btn share" onClick={share}>𝕏 {T("share")}</button>
        </div>
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
