// Açılış ekranı: ortada büyük robot kol kutudan parça alıp tepsiye bırakır; ardından kilit kendiliğinden
// döner ve iki çelik kapak mekanik olarak açılarak uygulamayı gösterir.
import { useEffect, useRef, useState } from "react";

type Phase = "pick" | "unlock" | "open";

/** Kol parçayı tepsiye bıraktıktan sonra (4 sn döngünün %78'i) kapaklar açılır. */
const PICK_MS = 3300;

export default function Splash({ onDone }: { onDone: () => void }) {
  const [phase, setPhase] = useState<Phase>("pick");
  const started = useRef(false);

  const enter = () => {
    if (started.current) return;
    started.current = true;
    setPhase("unlock");
    setTimeout(() => setPhase("open"), 750);
    setTimeout(onDone, 2100);
  };

  useEffect(() => {
    const t = setTimeout(enter, PICK_MS);
    const key = (e: KeyboardEvent) => { if (e.key === "Enter" || e.key === " " || e.key === "Escape") enter(); };
    window.addEventListener("keydown", key);
    return () => { clearTimeout(t); window.removeEventListener("keydown", key); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={`splash ${phase}`} role="dialog" aria-label="Açılış">
      <div className="door left"><DoorFace side="left" /></div>
      <div className="door right"><DoorFace side="right" /></div>
      <div className="lock" aria-hidden><div className="lock-ring" /><div className="lock-bar" /></div>

      <div className="splash-stage">
        <ArmScene />
      </div>
      <button className="splash-skip" onClick={enter}>Geç ⏎</button>
    </div>
  );
}

function DoorFace({ side }: { side: "left" | "right" }) {
  return (
    <>
      <div className="door-plate" />
      <div className={`door-stripe ${side}`} />
      <div className="door-rivets" />
      <div className="door-label">{side === "left" ? "HÜCRE 01" : "KAVRAMA"}</div>
    </>
  );
}

/** Üç eklemli robot kol: kutudan parçayı alır, kaldırır ve tepsiye bırakır (CSS anahtar kareleri). */
function ArmScene() {
  return (
    <svg className="arm-scene" viewBox="0 -30 600 450" aria-hidden>
      <defs>
        <linearGradient id="sp-metal" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#3a4654" />
          <stop offset="1" stopColor="#1d252f" />
        </linearGradient>
        <linearGradient id="sp-link" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffa15c" />
          <stop offset="1" stopColor="#d9600c" />
        </linearGradient>
      </defs>

      {/* zemin ızgarası */}
      <g className="sp-grid">
        {Array.from({ length: 13 }, (_, i) => <line key={`v${i}`} x1={i * 50} y1={360} x2={i * 50} y2={420} />)}
        <line x1={0} y1={360} x2={600} y2={360} />
        <line x1={0} y1={390} x2={600} y2={390} />
      </g>

      {/* tepsi (sol) */}
      <rect x={82} y={328} width={96} height={12} rx={3} fill="#2a3644" stroke="#4a5a6c" />
      <rect className="sp-part sp-tray-part" x={116} y={316} width={28} height={12} rx={2} />

      {/* kutu (sağ) ve yığılmış parçalar */}
      <g>
        <rect x={410} y={300} width={130} height={60} fill="#121921" stroke="#4a5a6c" strokeWidth={3} />
        <rect x={418} y={340} width={34} height={14} rx={2} className="sp-part dim" transform="rotate(-8 435 347)" />
        <rect x={492} y={338} width={30} height={16} rx={2} className="sp-part dim" transform="rotate(12 507 346)" />
        <rect x={446} y={326} width={22} height={18} rx={2} className="sp-part dim" />
        <rect className="sp-part sp-bin-part" x={456} y={330} width={28} height={12} rx={2} />
      </g>

      {/* taban */}
      <rect x={262} y={330} width={76} height={30} rx={6} fill="url(#sp-metal)" stroke="#4a5a6c" />
      <rect x={284} y={300} width={32} height={32} rx={4} fill="url(#sp-metal)" stroke="#4a5a6c" />

      {/* kol: omuz (300, 290) */}
      <g transform="translate(300 290)">
        <g className="sp-j1">
          <rect x={-10} y={-11} width={180} height={22} rx={11} fill="url(#sp-link)" />
          <circle r={17} fill="#1d252f" stroke="#ff7a1a" strokeWidth={3} />
          <circle r={5} fill="#ff7a1a" />
          <g transform="translate(160 0)">
            <g className="sp-j2">
              <rect x={-9} y={-9} width={158} height={18} rx={9} fill="url(#sp-link)" />
              <circle r={14} fill="#1d252f" stroke="#ff7a1a" strokeWidth={3} />
              <circle r={4} fill="#ff7a1a" />
              <g transform="translate(140 0)">
                <g className="sp-j3">
                  <circle r={10} fill="#1d252f" stroke="#9aa6b2" strokeWidth={2} />
                  {/* bilek aşağı (+x yönünde) bakar; tutucu gövdesi */}
                  <rect x={6} y={-12} width={14} height={24} rx={3} fill="#2c3846" stroke="#9aa6b2" />
                  <rect className="sp-finger top" x={20} y={-3} width={22} height={5} rx={2} fill="#c9d1d9" />
                  <rect className="sp-finger bottom" x={20} y={-2} width={22} height={5} rx={2} fill="#c9d1d9" />
                  <rect className="sp-part sp-held-part" x={26} y={-14} width={12} height={28} rx={2} />
                </g>
              </g>
            </g>
          </g>
        </g>
      </g>

      {/* kamera ve görüş konisi */}
      <g className="sp-cam">
        <polygon points="475,40 410,300 540,300" fill="rgba(88,166,255,0.06)" />
        <rect x={460} y={24} width={30} height={18} rx={4} fill="#1d252f" stroke="#58a6ff" />
        <circle cx={475} cy={42} r={4} fill="#58a6ff" />
      </g>
    </svg>
  );
}
