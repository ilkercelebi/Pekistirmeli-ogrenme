import katex from "katex";
import { useMemo, type ReactNode } from "react";
import type { FeatureValue, GraspEvent, Outcome, Tone } from "../api";
import { C, FEATURE_TITLE, OUTCOME, TONE } from "../theme";

export function PageHead({ kicker, title, children }: { kicker: string; title: string; children?: ReactNode }) {
  return (
    <header className="page-head fade-in">
      <div className="kicker">{kicker}</div>
      <h1>{title}</h1>
      {children && <p>{children}</p>}
    </header>
  );
}

export function Section({ num, title, children }: { num: string | number; title: string; children?: ReactNode }) {
  return (
    <div className="sec">
      <span className="num">{num}</span>
      <div>
        <h2>{title}</h2>
        {children && <p>{children}</p>}
      </div>
    </div>
  );
}

export function Card({
  title, kicker, tone, children, className = "",
}: { title?: string; kicker?: string; tone?: string; children?: ReactNode; className?: string }) {
  const color = tone ?? C.accent;
  return (
    <div className={`card ${tone ? "top" : ""} ${className}`} style={tone ? { borderTopColor: color } : undefined}>
      {kicker && <div className="card-k" style={{ color }}>{kicker}</div>}
      {title && <h3>{title}</h3>}
      <div className="body">{children}</div>
    </div>
  );
}

export function Stat({ label, value, sub, color }: { label: string; value: ReactNode; sub?: string; color?: string }) {
  return (
    <div className="stat">
      <div className="l">{label}</div>
      <div className="v" style={{ color: color ?? C.text }}>{value}</div>
      {sub && <div className="s">{sub}</div>}
    </div>
  );
}

export function Chip({ children, tone, color, name }: { children: ReactNode; tone?: Tone; color?: string; name?: string }) {
  const c = color ?? (tone ? TONE[tone] : C.info);
  return (
    <span className="chip" style={{ color: c, borderColor: `${c}55`, background: `${c}14` }}>
      {name && <span className="n">{name}</span>}
      {children}
    </span>
  );
}

export function FeatureChips({ features }: { features: FeatureValue[] }) {
  return (
    <div>
      {features.map((f) => (
        <Chip key={f.name} tone={f.tone} name={FEATURE_TITLE[f.name] ?? f.name}>{f.label}</Chip>
      ))}
    </div>
  );
}

export function Banner({ event }: { event: GraspEvent | null }) {
  if (!event) {
    return (
      <div className="banner idle">
        <div>
          <div className="o">HAZIR</div>
          <div className="r">Henüz kavrama yapılmadı.</div>
        </div>
      </div>
    );
  }
  const o = OUTCOME[event.outcome];
  return (
    <div className="banner fade-in" style={{ borderColor: `${o.color}66`, background: `${o.color}12` }}>
      <div>
        <div className="o" style={{ color: o.color }}>{o.label}</div>
        <div className="r">
          {event.reason} · θ = {Math.round(event.action.theta_deg ?? 0)}° · ({event.action.col}, {event.action.row})
        </div>
      </div>
      <div className="w" style={{ color: o.color }}>{event.reward > 0 ? "+" : ""}{event.reward.toFixed(1)}</div>
    </div>
  );
}

export function Timeline({ outcomes, total }: { outcomes: Outcome[]; total?: number }) {
  const empty = Math.max(0, (total ?? 0) - outcomes.length);
  return (
    <div className="timeline">
      {outcomes.map((o, i) => (
        <span key={i} style={{ background: OUTCOME[o].color }} title={OUTCOME[o].label}>{i + 1}</span>
      ))}
      {Array.from({ length: Math.min(empty, 40) }, (_, i) => <span key={`e${i}`} className="empty" />)}
    </div>
  );
}

export function OutcomeLegend() {
  return (
    <div className="legend">
      {(Object.keys(OUTCOME) as Outcome[]).map((k) => (
        <span key={k}><i style={{ background: OUTCOME[k].color }} />{OUTCOME[k].label.toLowerCase()} ({OUTCOME[k].reward})</span>
      ))}
    </div>
  );
}

export function TeX({ children, block = true }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: block, throwOnError: false }), [children, block]);
  if (!block) return <span dangerouslySetInnerHTML={{ __html: html }} />;
  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Seg<T extends string | number>({
  value, options, onChange, disabled,
}: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; disabled?: boolean }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={String(o.value)} className={o.value === value ? "on" : ""} disabled={disabled} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty-state">
      <b>{title}</b>
      {children}
    </div>
  );
}
