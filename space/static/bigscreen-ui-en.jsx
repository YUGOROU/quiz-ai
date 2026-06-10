// Quiz Buzzer AI big-screen display — UI components (English)
// Presentational only. State arrives from the engine via props.

// Split the question text into seen / buzz-marker / unseen.
// aiPlan: 人間が先押しした問題の判定後に「AIはここで押す予定だった」位置（frac）を示す。
function QuestionText({ q, seen, phase, buzzer, accentLabel, revealRest, aiPlan }) {
  const full = q.full;
  const seenStr = full.slice(0, seen);
  const restStr = full.slice(seen);
  const reading = phase === "reading";
  const buzzed = buzzer && (phase === "buzzed" || phase === "answer" || phase === "judged");
  const aiChar = aiPlan ? Math.round(aiPlan * full.length) : 0;
  const showPlan = !!(aiPlan && revealRest && aiChar > seen && aiChar < full.length);
  const tailCls = revealRest ? "tail" : "unseen";
  return (
    <div className="qtext">
      <span className="seen">{seenStr}</span>
      {buzzed && (
        <span className="buzzmark" data-label={accentLabel}></span>
      )}
      {reading && <span className="caret"></span>}
      {showPlan ? (
        <span className={tailCls}>
          {full.slice(seen, aiChar)}
          <span className="buzzmark plan"
            data-label={T("aiPlanned") + Math.round(aiPlan * 100) + "%"}></span>
          {full.slice(aiChar)}
        </span>
      ) : (
        <span className={tailCls}>{restStr}</span>
      )}
    </div>
  );
}

function TopBar({ qIndex, total, phase, genre }) {
  const phaseLabel = T("phase." + phase) || phase;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="dot"></span>
        Quiz Buzzer AI
      </div>
      {genre ? <span className="genre-tag">{genre}</span> : null}
      <div className="spacer"></div>
      <span className="qno">{T("q")} <b>{Math.min(qIndex + 1, total)}</b> / {total} {T("qof")}</span>
      <span className="phase-tag" data-p={phase === "buzzed" ? "buzz" : ""}>{phaseLabel}</span>
    </header>
  );
}

function AnswerCard({ side, answer, result, show }) {
  const cls = ["answer-card", show ? "show" : "", result ? (result.correct ? "ok" : "ng") : ""].join(" ");
  return (
    <div className={cls}>
      <span className="al">{side === "ai" ? T("ai.answer") : T("answer")}</span>
      <span className="av">{answer || "\u2014"}</span>
      {result && (
        <span className={"verdict " + (result.correct ? "ok" : "ng")}>
          {result.correct ? T("correct") : T("wrong")}
        </span>
      )}
    </div>
  );
}

function AIColumn({ q, thinkShown, phase, buzzer, result, showReasoning, confidence, theta }) {
  const aiAnswered = buzzer === "ai" && (phase === "answer" || phase === "judged");
  const live = phase === "reading";
  // 判定が出るまで think は masked（答えを ●● 化）で表示する。読めばカンニングできてしまうため。
  const revealed = phase === "judged" || phase === "roundover";
  const thinkText = (t) => (revealed ? t.text : (t.masked != null ? t.masked : t.text));
  const confPct = Math.max(0, Math.min(100, Math.round((confidence || 0) * 100)));
  const thetaPct = Math.max(0, Math.min(100, Math.round((theta || 0) * 100)));
  return (
    <section className="col ai" style={{ "--accent": "var(--ai)" }}>
      <div className="side-head">
        <div className="avatar">AI</div>
        <div className="who">{T("buzzerAI")}</div>
      </div>
      <div className="status-line">
        <span className={"status-dot" + (live ? " live" : "")}></span>
        {live ? T("reasoningDots") : aiAnswered ? T("answered") : phase === "buzzed" && buzzer === "ai" ? T("buzzedIn") : T("standby")}
      </div>

      {/* buzz回帰ヘッドの実測確信度（confCurve 再生）。θ に迫る＝AIが押す予兆。 */}
      <div className="confwrap">
        <div className="confbar">
          <div className="cb-fill" style={{ width: confPct + "%" }}></div>
          <div className="cb-th" style={{ left: thetaPct + "%" }}></div>
        </div>
        <div className="cb-lab">{T("confLabel")} <b>{confPct}%</b><span className="cb-theta">θ {thetaPct}%</span></div>
      </div>

      <div className="col-mid">
        {showReasoning && (
          <div className="reason">
            <div className="rh">{T("reasoning")}</div>
            <ul className="think-list">
              {thinkShown.map((t, i) => (
                <li key={i} className={i === thinkShown.length - 1 ? "cur" : ""}>{thinkText(t)}</li>
              ))}
            </ul>
          </div>
        )}
        {aiAnswered && <AnswerCard side="ai" answer={q.answer} result={result} show={true} />}
      </div>
    </section>
  );
}

function AnswerInput({ onSubmit, onPass, timeLeft }) {
  const [val, setVal] = React.useState("");
  const [listening, setListening] = React.useState(false);
  const [err, setErr] = React.useState("");
  const recRef = React.useRef(null);
  const inputRef = React.useRef(null);
  const SR = (typeof window !== "undefined") && (window.SpeechRecognition || window.webkitSpeechRecognition);

  React.useEffect(() => {
    if (inputRef.current) inputRef.current.focus();
    return () => { if (recRef.current) { try { recRef.current.stop(); } catch (e) {} } };
  }, []);

  const toggleMic = () => {
    if (!SR) { setErr(T("voiceUnsupported")); return; }
    if (listening) { try { recRef.current && recRef.current.stop(); } catch (e) {} return; }
    const rec = new SR();
    // \u554f\u984c\u306f\u65e5\u672c\u8a9e\u56fa\u5b9a\u306a\u306e\u3067\u3001UI \u8a00\u8a9e\u306b\u95a2\u308f\u3089\u305a\u89e3\u7b54\u306e\u97f3\u58f0\u8a8d\u8b58\u306f\u65e5\u672c\u8a9e\u3002
    rec.lang = "ja-JP"; rec.interimResults = true; rec.maxAlternatives = 1; rec.continuous = false;
    rec.onresult = (e) => {
      let t = "";
      for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
      setVal(t);
    };
    rec.onerror = (e) => {
      setListening(false);
      if (e.error === "not-allowed" || e.error === "service-not-allowed")
        setErr(T("micDenied"));
    };
    rec.onend = () => { setListening(false); recRef.current = null; };
    recRef.current = rec; setErr(""); setListening(true);
    try { rec.start(); } catch (e) { setListening(false); }
  };

  const submit = () => {
    const t = val.trim();
    if (!t) return;
    if (recRef.current) { try { recRef.current.stop(); } catch (e) {} }
    onSubmit(t);
  };

  return (
    <div className="answer-input">
      <div className="ai-label">
        {T("yourAnswer")}
        {timeLeft != null && (
          <span className={"ai-timer" + (timeLeft <= 3 ? " urgent" : "")}>
            {T("timeLeft")} {timeLeft}s
          </span>
        )}
      </div>
      <div className="ai-row">
        <input ref={inputRef} className="ai-field" type="text" value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => {
            // IME 変換確定の Enter で送信しない（isComposing / keyCode 229 を除外）。
            if (e.key === "Enter" && !e.nativeEvent.isComposing && e.keyCode !== 229) {
              e.preventDefault(); submit();
            }
          }}
          placeholder={T("typePlaceholder")} autoComplete="off" spellCheck="false" />
        <button type="button" className={"ai-mic" + (listening ? " on" : "")}
          onClick={toggleMic} title="Voice input" aria-label="Voice input">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor" />
            <path d="M6 11a6 6 0 0012 0M12 17v3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      <div className="ai-actions">
        {listening ? (
          <span className="ai-listen"><span className="lw"><i></i><i></i><i></i></span>{T("listening")}</span>
        ) : (
          <span className="ai-hint">{SR ? T("micHint") : T("typeHint")}</span>
        )}
        <span className="ai-spacer"></span>
        <button type="button" className="ai-pass" onClick={onPass}>{T("pass")}</button>
        <button type="button" className="ai-submit" onClick={submit} disabled={!val.trim()}>{T("submit")}</button>
      </div>
      {err && <div className="ai-err">{err}</div>}
    </div>
  );
}

function HumanColumn({ q, phase, buzzer, result, followUp, humanAnswer, timeLeft, onBuzz, onAnswer, onPass }) {
  const live = phase === "reading";
  const answering = phase === "answering" && buzzer === "human";
  const humanAnswered = (buzzer === "human" && (phase === "answer" || phase === "judged"))
    || (phase === "judged" && followUp);
  // humanAnswer が空＝パス。q.answer（AIの解答）にフォールバックしてはいけない。
  const ans = buzzer === "human" ? (humanAnswer || T("pass")) : (followUp ? followUp.answer : null);
  const res = buzzer === "human" ? result : (followUp ? { correct: followUp.correct } : null);
  return (
    <section className="col human" style={{ "--accent": "var(--hu)" }}>
      <div className="side-head">
        <div className="avatar">{T("you")}</div>
        <div className="who">{T("player")}</div>
      </div>
      <div className="status-line">
        <span className={"status-dot" + (live ? " live" : "")}></span>
        {live ? T("listening") : answering ? T("yourTurn") : humanAnswered ? T("answered")
          : phase === "buzzed" && buzzer === "human" ? T("buzzedIn") : T("standby")}
      </div>

      <div className="col-mid human-body">
        {answering ? (
          <AnswerInput onSubmit={onAnswer} onPass={onPass} timeLeft={timeLeft} />
        ) : !humanAnswered ? (
          <React.Fragment>
            <div className={"buzzer-ready" + (live ? " armed" : "") + ((phase === "buzzed" && buzzer === "human") ? " hit" : "")}
              role="button" tabIndex={live ? 0 : -1}
              onClick={live ? onBuzz : undefined}
              onKeyDown={live ? ((e) => { if (e.key === "Enter") onBuzz(); }) : undefined}>
              <div className="br-ring"></div>
              <div className="br-core">
                {phase === "buzzed" && buzzer === "human" ? T("buzz") : live ? T("buzz") : "\u2014"}
              </div>
            </div>
            {live && (
              <div className="buzz-hint">{T("buzzHint.pre")}<kbd>Space</kbd>{T("buzzHint.post")}</div>
            )}
            {phase === "buzzed" && buzzer === "human" && (
              <div className="mic">
                <span className="wave"><i></i><i></i><i></i><i></i><i></i></span>
                {T("transcribing")}
              </div>
            )}
          </React.Fragment>
        ) : (
          <AnswerCard side="human" answer={ans} result={res} show={true} />
        )}
      </div>
    </section>
  );
}

function CenterBoard({ scores }) {
  // 主表示はポイント（早押しボーナス 1.0〜1.5・誤答 −1.5 込み）。
  // 正解数だけだと「ギリギリまで聞いて押す」が常に最適になり、早押しの意味が消えるため。
  const fmt = (v) => (Math.round(v * 10) / 10).toFixed(1);
  return (
    <aside className="center">
      <div className="score-grid">
        <div className="score-num ai">{fmt(scores.ai.pts)}
          <span className="score-sub">{scores.ai.correct} {T("correctN")}</span>
          <small>{T("ai")}</small></div>
        <div className="vs">{T("vs")}</div>
        <div className="score-num hu">{fmt(scores.human.pts)}
          <span className="score-sub">{scores.human.correct} {T("correctN")}</span>
          <small>{T("human")}</small></div>
      </div>
      <div className="score-unit">{T("pts")}</div>
    </aside>
  );
}

function FlashLayer({ flash, fx }) {
  if (!flash || !flash.on) return null;
  const accent = flash.side === "ai" ? "var(--ai)" : "var(--hu)";
  const label = flash.side === "ai" ? T("aiBuzz") : T("humanBuzz");
  const dur = fx === "flashy" ? "1.05s" : "0.7s";
  const str = fx === "flashy" ? "52%" : "30%";
  return (
    <React.Fragment>
      <div className={"siderail " + (flash.side === "ai" ? "l" : "r") + " on"} style={{ "--accent": accent }}></div>
      <div className="flash on" key={flash.key} style={{ "--accent": accent, "--flashdur": dur, "--flashstr": str }}>
        <div className="bz-label">{label}</div>
      </div>
    </React.Fragment>
  );
}

Object.assign(window, { QuestionText, TopBar, AIColumn, HumanColumn, CenterBoard, FlashLayer, AnswerCard });
