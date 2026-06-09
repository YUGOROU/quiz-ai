// Quiz Buzzer AI big-screen display — styles (English)
// 左右対決の構図と早押し色演出は維持。ゲージ/統計/タイミング比較/テレメトリ/メタ表記は削除。
// 方向性は [data-dir]（competitive=クリーン / broadcast=TV映え）、演出強度は [data-fx]。
// 配色アクセントは CSS変数 --hu(人間) / --ai で注入。
window.BIGSCREEN_CSS = `
:root{
  --bg:#f1efe9; --panel:#fbfaf7; --panel2:#f6f4ee;
  --ink:#1c1b18; --ink2:#5d5a52; --ink3:#94908691;
  --line:#e0ddd3; --line2:#ebe8df;
  --hu:#e0453f; --ai:#2f6df0;
  --ai-soft:color-mix(in oklab, var(--ai) 14%, var(--panel));
  --ok:#1f9d57; --ng:#d4342c;
  --q-font:"Newsreader",Georgia,serif;
  --ui-font:"Schibsted Grotesk",system-ui,sans-serif;
  --mono:"DM Mono",ui-monospace,monospace;
  --r:16px;
}
*{box-sizing:border-box;margin:0;padding:0}
.stage-root{width:1920px;height:1080px;position:relative;overflow:hidden;
  background:
    radial-gradient(120% 80% at 50% -10%, color-mix(in oklab,var(--ai) 5%,var(--bg)), var(--bg) 55%),
    var(--bg);
  color:var(--ink);font-family:var(--ui-font);
  display:grid;grid-template-rows:80px 1fr 264px;}

/* ── TOP BAR（最小限） ──────────────────────────────── */
.topbar{display:flex;align-items:center;gap:24px;padding:0 48px;
  border-bottom:1px solid var(--line);background:var(--panel);}
.brand{display:flex;align-items:center;gap:13px;font-weight:800;font-size:22px;letter-spacing:.01em;white-space:nowrap;}
.brand .dot{width:12px;height:12px;border-radius:4px;background:var(--ai);
  box-shadow:0 0 0 5px color-mix(in oklab,var(--ai) 20%,transparent);}
.topbar .spacer{flex:1;}
.qno{font-family:var(--mono);font-size:17px;color:var(--ink2);letter-spacing:.02em;}
.qno b{color:var(--ink);font-variant-numeric:tabular-nums;font-size:20px;}
.phase-tag{font-size:15px;font-weight:700;padding:8px 18px;border-radius:999px;
  letter-spacing:.04em;background:var(--ink);color:var(--bg);transition:.25s;}
.phase-tag[data-p="buzz"]{background:var(--accent,var(--ink));}

/* ── MID: 3 columns（対決の構図） ───────────────────── */
.mid{display:grid;grid-template-columns:1fr 400px 1fr;min-height:0;}
.col{display:flex;flex-direction:column;min-height:0;position:relative;padding:36px 44px;gap:22px;}
.col.ai{border-right:1px solid var(--line);}
.col.human{border-left:1px solid var(--line);}

.side-head{display:flex;align-items:center;gap:16px;}
.avatar{width:60px;height:60px;border-radius:16px;display:grid;place-items:center;
  font-weight:800;font-size:24px;color:#fff;flex:none;letter-spacing:.02em;}
.col.ai .avatar{background:linear-gradient(150deg,var(--ai),color-mix(in oklab,var(--ai) 60%,#000));}
.col.human .avatar{background:linear-gradient(150deg,var(--hu),color-mix(in oklab,var(--hu) 60%,#000));}
.side-head .who{font-weight:800;font-size:26px;line-height:1.05;white-space:nowrap;}
.col.human .side-head{flex-direction:row-reverse;text-align:right;}

.status-line{display:flex;align-items:center;gap:10px;font-size:16px;color:var(--ink2);font-weight:600;}
.col.human .status-line{flex-direction:row-reverse;}
.status-dot{width:10px;height:10px;border-radius:50%;background:var(--ink3);flex:none;transition:.2s;}
.status-dot.live{background:var(--accent);animation:ping 1.4s infinite;}
@keyframes ping{0%{box-shadow:0 0 0 0 color-mix(in oklab,var(--accent) 55%,transparent);}
  70%{box-shadow:0 0 0 11px transparent;}100%{box-shadow:0 0 0 0 transparent;}}

/* centered middle stack for each side column */
.col-mid{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;gap:18px;}
.col.human .col-mid{align-items:flex-end;}

/* AI 思考プロセス（reasoning は維持・メタ装飾は削除） */
.reason{flex:0 1 auto;max-height:440px;display:flex;flex-direction:column;gap:14px;
  background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:22px 24px;}
.reason .rh{font-size:13px;letter-spacing:.04em;color:var(--ink2);font-weight:600;}
.think-list{list-style:none;display:flex;flex-direction:column;gap:12px;overflow:hidden;}
.think-list li{font-size:16px;line-height:1.55;color:var(--ink2);padding-left:20px;position:relative;}
.think-list li::before{content:"›";position:absolute;left:2px;color:var(--ai);font-weight:700;}
.think-list li.cur{color:var(--ink);font-weight:600;}

/* human 早押し待機 / BUZZ */
.human-body{display:flex;flex-direction:column;gap:20px;align-items:flex-end;justify-content:center;text-align:right;}
.buzzer-ready{position:relative;width:208px;height:208px;display:grid;place-items:center;margin:0 auto;}
.buzzer-ready .br-ring{position:absolute;inset:0;border-radius:50%;
  border:2px solid color-mix(in oklab,var(--hu) 30%,var(--line));}
.buzzer-ready .br-core{width:156px;height:156px;border-radius:50%;display:grid;place-items:center;
  font:800 30px/1 var(--ui-font);letter-spacing:.08em;color:var(--ink3);
  background:var(--panel);border:1px solid var(--line);transition:.3s;}
.buzzer-ready.armed .br-ring{border-color:color-mix(in oklab,var(--hu) 55%,transparent);
  animation:brpulse 1.8s ease-out infinite;}
.buzzer-ready.armed .br-core{color:var(--hu);
  box-shadow:0 0 0 7px color-mix(in oklab,var(--hu) 8%,transparent),0 12px 34px color-mix(in oklab,var(--hu) 16%,transparent);}
.buzzer-ready.hit .br-core{color:#fff;background:var(--hu);border-color:var(--hu);transform:scale(1.04);
  box-shadow:0 0 0 12px color-mix(in oklab,var(--hu) 18%,transparent),0 16px 50px color-mix(in oklab,var(--hu) 40%,transparent);}
.buzzer-ready.hit .br-ring{border-color:var(--hu);}
@keyframes brpulse{0%{box-shadow:0 0 0 0 color-mix(in oklab,var(--hu) 35%,transparent);}
  70%{box-shadow:0 0 0 30px transparent;}100%{box-shadow:0 0 0 0 transparent;}}
.stage-root[data-fx="calm"] .buzzer-ready.armed .br-ring{animation-duration:3s;}
.mic{display:flex;align-items:center;gap:11px;color:var(--ink2);font-size:16px;font-weight:600;}
.mic .wave{display:flex;gap:3px;align-items:center;height:24px;}
.mic .wave i{width:4px;height:8px;background:var(--hu);border-radius:2px;animation:wv 1s infinite ease-in-out;}
.mic .wave i:nth-child(2){animation-delay:.15s}.mic .wave i:nth-child(3){animation-delay:.3s}
.mic .wave i:nth-child(4){animation-delay:.45s}.mic .wave i:nth-child(5){animation-delay:.6s}
@keyframes wv{0%,100%{height:7px}50%{height:22px}}

/* 回答カード（正誤の色のみ。ポイント増減は中央スコアに集約） */
.answer-card{border-radius:var(--r);padding:22px 24px;border:1px solid var(--line);background:var(--panel);
  display:flex;flex-direction:column;gap:10px;opacity:0;transform:translateY(8px);transition:.3s;}
.answer-card.show{opacity:1;transform:none;}
.answer-card .al{font-size:13px;letter-spacing:.04em;color:var(--ink2);font-weight:600;}
.answer-card .av{font-family:var(--q-font);font-size:40px;font-weight:700;line-height:1.05;}
.col.human .answer-card{align-items:flex-end;text-align:right;}
.answer-card.ok{border-color:color-mix(in oklab,var(--ok) 45%,var(--line));background:color-mix(in oklab,var(--ok) 8%,var(--panel));}
.answer-card.ng{border-color:color-mix(in oklab,var(--ng) 45%,var(--line));background:color-mix(in oklab,var(--ng) 8%,var(--panel));}
.verdict{display:inline-flex;align-items:center;gap:8px;font-weight:800;font-size:19px;}
.verdict.ok{color:var(--ok);}.verdict.ng{color:var(--ng);}

/* ── CENTER：スコアのみ ───────────────────────────── */
.center{background:var(--panel);border-left:1px solid var(--line);border-right:1px solid var(--line);
  display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;position:relative;}
.score-grid{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;width:100%;}
.score-num{font-weight:800;font-size:104px;line-height:.86;font-variant-numeric:tabular-nums;letter-spacing:-.02em;text-align:center;}
.score-num.ai{color:var(--ai);}.score-num.hu{color:var(--hu);}
.score-num small{display:block;font-size:15px;font-weight:700;letter-spacing:.08em;
  color:var(--ink2);margin-top:14px;}
.vs{font-weight:800;font-size:26px;color:var(--ink3);font-family:var(--mono);}

/* ── QUESTION BAND：問題文のみ ─────────────────────── */
.qband{border-top:1px solid var(--line);background:var(--panel);padding:0 64px;
  display:flex;align-items:center;position:relative;}
.qtext{font-family:var(--q-font);font-size:52px;line-height:1.42;font-weight:600;letter-spacing:.01em;
  position:relative;text-wrap:pretty;width:100%;}
.qtext .seen{color:var(--ink);}
.qtext .caret{display:inline-block;width:4px;height:.95em;background:var(--ai);vertical-align:-12%;
  margin-left:3px;animation:blink 1s step-end infinite;border-radius:2px;}
@keyframes blink{50%{opacity:0;}}
.qtext .unseen{color:transparent;}
.qtext .buzzmark{display:inline-block;position:relative;}
.qtext .buzzmark::before{content:"";position:absolute;left:-2px;top:-8px;bottom:-8px;width:4px;
  background:var(--accent);border-radius:2px;box-shadow:0 0 12px var(--accent);}
.qtext .buzzmark::after{content:attr(data-label);position:absolute;left:-2px;top:-36px;
  font-family:var(--mono);font-size:15px;font-weight:600;color:#fff;background:var(--accent);
  padding:3px 11px;border-radius:7px;white-space:nowrap;letter-spacing:.02em;}
.qtext .tail{color:var(--ink3);font-style:normal;}

/* ── FLASH OVERLAY（早押し色演出・維持） ──────────── */
.flash{position:absolute;inset:0;pointer-events:none;z-index:40;opacity:0;}
.flash.on{animation:flashfx var(--flashdur,.9s) ease-out;}
.flash::before{content:"";position:absolute;inset:0;
  background:radial-gradient(110% 70% at 50% 50%, color-mix(in oklab,var(--accent) var(--flashstr,42%),transparent), transparent 70%);}
@keyframes flashfx{0%{opacity:0;}12%{opacity:1;}100%{opacity:0;}}
.flash .bz-label{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%) scale(.7);
  font-family:var(--ui-font);font-weight:900;font-size:150px;letter-spacing:.04em;color:var(--accent);
  opacity:0;text-shadow:0 6px 40px color-mix(in oklab,var(--accent) 40%,transparent);}
.flash.on .bz-label{animation:bzpop var(--flashdur,.9s) cubic-bezier(.2,.9,.3,1);}
@keyframes bzpop{0%{opacity:0;transform:translate(-50%,-50%) scale(.6);}
  20%{opacity:1;transform:translate(-50%,-50%) scale(1.06);}
  70%{opacity:1;transform:translate(-50%,-50%) scale(1);}100%{opacity:0;transform:translate(-50%,-50%) scale(1);}}
.siderail{position:absolute;top:0;bottom:0;width:16px;z-index:35;opacity:0;background:var(--accent);
  transition:opacity .2s;box-shadow:0 0 44px var(--accent);}
.siderail.l{left:0;}.siderail.r{right:0;}
.siderail.on{opacity:.9;}

/* ── direction: BROADCAST（TV映え／より大胆） ───────── */
.stage-root[data-dir="broadcast"]{--r:20px;
  background:radial-gradient(130% 90% at 50% -20%, color-mix(in oklab,var(--ai) 12%,var(--bg)), var(--bg) 60%);}
.stage-root[data-dir="broadcast"] .qtext{font-size:58px;font-weight:700;}
.stage-root[data-dir="broadcast"] .score-num{font-size:120px;
  text-shadow:0 4px 30px color-mix(in oklab,currentColor 28%,transparent);}
.stage-root[data-dir="broadcast"] .avatar{box-shadow:0 10px 30px rgba(0,0,0,.18);width:66px;height:66px;font-size:26px;}
.stage-root[data-dir="broadcast"] .side-head .who{font-size:30px;}
.stage-root[data-dir="broadcast"] .reason,
.stage-root[data-dir="broadcast"] .answer-card{box-shadow:0 12px 38px rgba(40,38,30,.08);}
.stage-root[data-dir="broadcast"] .center{background:linear-gradient(180deg,var(--panel),var(--panel2));}

/* ── intensity caps ────────────────────────────────── */
.stage-root[data-fx="calm"] .flash .bz-label{display:none;}
.stage-root[data-fx="calm"] .siderail{width:9px;}
.stage-root[data-fx="calm"] .status-dot.live{animation-duration:2s;}

/* ── LIVE HUMAN INPUT (English build): text + voice ─────── */
.buzzer-ready[role="button"]{cursor:default;outline:none;}
.buzzer-ready.armed[role="button"]{cursor:pointer;}
.buzzer-ready.armed[role="button"]:hover .br-core{transform:scale(1.03);}
.buzz-hint{margin:0 auto;font-size:16px;color:var(--ink2);font-weight:500;text-align:center;}
.buzz-hint kbd{font:600 12px var(--mono);background:var(--panel2);border:1px solid var(--line);
  border-bottom-width:2px;border-radius:6px;padding:2px 8px;color:var(--ink);margin:0 1px;}

.answer-input{width:440px;max-width:100%;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r);padding:22px 22px 18px;display:flex;flex-direction:column;gap:13px;text-align:left;
  box-shadow:0 16px 44px color-mix(in oklab,var(--hu) 12%,transparent);
  animation:aiIn .28s cubic-bezier(.2,.9,.3,1);}
@keyframes aiIn{from{transform:translateY(10px);}to{transform:none;}}
.answer-input .ai-label{font-size:13px;letter-spacing:.04em;color:var(--ink2);font-weight:600;}
.ai-row{display:flex;gap:10px;align-items:stretch;}
.ai-field{flex:1;min-width:0;font:600 28px var(--q-font);color:var(--ink);background:var(--panel2);
  border:1.5px solid var(--line);border-radius:12px;padding:12px 16px;outline:none;transition:.18s;}
.ai-field:focus{border-color:var(--hu);background:var(--panel);
  box-shadow:0 0 0 4px color-mix(in oklab,var(--hu) 14%,transparent);}
.ai-field::placeholder{color:var(--ink3);font-weight:500;}
.ai-mic{flex:none;width:58px;border-radius:12px;border:1.5px solid var(--line);background:var(--panel2);
  color:var(--ink2);display:grid;place-items:center;cursor:pointer;transition:.18s;}
.ai-mic:hover{color:var(--hu);border-color:color-mix(in oklab,var(--hu) 45%,var(--line));}
.ai-mic.on{background:var(--hu);border-color:var(--hu);color:#fff;animation:micpulse 1.4s infinite;}
@keyframes micpulse{0%{box-shadow:0 0 0 0 color-mix(in oklab,var(--hu) 42%,transparent);}
  70%{box-shadow:0 0 0 15px transparent;}100%{box-shadow:0 0 0 0 transparent;}}
.ai-actions{display:flex;align-items:center;gap:10px;}
.ai-spacer{flex:1;}
.ai-hint,.ai-listen{font-size:14px;color:var(--ink2);font-weight:500;display:flex;align-items:center;gap:8px;}
.ai-listen{color:var(--hu);font-weight:700;}
.ai-listen .lw{display:flex;gap:3px;align-items:center;height:18px;}
.ai-listen .lw i{width:3px;height:7px;background:var(--hu);border-radius:2px;animation:wv 1s infinite ease-in-out;}
.ai-listen .lw i:nth-child(2){animation-delay:.15s}.ai-listen .lw i:nth-child(3){animation-delay:.3s}
.ai-pass{font:600 15px var(--ui-font);color:var(--ink2);background:transparent;border:1.5px solid var(--line);
  border-radius:999px;padding:9px 18px;cursor:pointer;transition:.18s;}
.ai-pass:hover{color:var(--ink);border-color:var(--ink3);}
.ai-submit{font:700 15px var(--ui-font);color:#fff;background:var(--hu);border:1.5px solid var(--hu);
  border-radius:999px;padding:9px 24px;cursor:pointer;transition:.18s;}
.ai-submit:hover{filter:brightness(1.06);}
.ai-submit:disabled{opacity:.4;cursor:not-allowed;}
.ai-err{font-size:13px;color:var(--ng);font-weight:600;}
`;
