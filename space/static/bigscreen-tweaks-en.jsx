// Quiz Buzzer AI big-screen — Tweaks panel (English)
// useTweaks manages state and bridges changes to BigScreenApp via
// window.__twState + window.__onTweaks (across separate <script> scopes).
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "direction": "competitive",
  "intensity": "flashy",
  "reasoning": true,
  "speed": 1,
  "humanColor": "#e0453f",
  "aiColor": "#2f6df0"
}/*EDITMODE-END*/;

function BigScreenTweaks() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  // Bridge: notify the app on every tweak change
  React.useEffect(() => {
    window.__twState = t;
    if (window.__onTweaks) window.__onTweaks(t);
    // accent colors → CSS variables
    const root = document.querySelector(".stage-root");
    if (root) {
      root.style.setProperty("--hu", t.humanColor);
      root.style.setProperty("--ai", t.aiColor);
    }
  }, [t]);

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Visual direction" />
      <TweakRadio label="Tone" value={t.direction}
        options={[
          { value: "competitive", label: "Clean" },
          { value: "broadcast", label: "Broadcast" },
        ]}
        onChange={(v) => setTweak("direction", v)} />

      <TweakRadio label="FX intensity" value={t.intensity}
        options={[{ value: "flashy", label: "Bold" }, { value: "calm", label: "Calm" }]}
        onChange={(v) => setTweak("intensity", v)} />

      <TweakSection label="Display" />
      <TweakToggle label="Show AI reasoning" value={t.reasoning}
        onChange={(v) => setTweak("reasoning", v)} />
      <TweakSlider label="Playback speed" value={t.speed} min={0.5} max={2} step={0.25} unit="×"
        onChange={(v) => setTweak("speed", v)} />

      <TweakSection label="Accent" />
      <TweakColor label="Human" value={t.humanColor}
        options={["#e0453f", "#dc2626", "#ea580c", "#db2777"]}
        onChange={(v) => setTweak("humanColor", v)} />
      <TweakColor label="AI" value={t.aiColor}
        options={["#2f6df0", "#0ea5e9", "#7c3aed", "#0d9488"]}
        onChange={(v) => setTweak("aiColor", v)} />
    </TweaksPanel>
  );
}

// Mount (independent root)
(function () {
  const el = document.createElement("div");
  document.body.appendChild(el);
  ReactDOM.createRoot(el).render(<BigScreenTweaks />);
})();
