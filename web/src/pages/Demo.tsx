import { useEffect, useMemo, useRef, useState } from "react";
import { api, type GraspAction, type GripperKind, type Preview } from "../api";
import GridCanvas, { type Footprint } from "../components/GridCanvas";
import { Banner, Card, Empty, FeatureChips, Field, OutcomeLegend, PageHead, Seg, Stat, Timeline } from "../components/ui";
import { useMeta, useMethods, useSimulation } from "../hooks";
import RobotCell from "../three/Scene";
import { C, OUTCOME, fmt } from "../theme";

export default function Demo() {
  const [gripper, setGripper] = useState<GripperKind>("parallel_jaw");
  const methods = useMethods(gripper);
  const [method, setMethod] = useState("heuristic");
  const [seed, setSeed] = useState(7);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [manual, setManual] = useState(false);
  const [angle, setAngle] = useState(0);
  const [overlay, setOverlay] = useState<"none" | "q" | "depth">("none");
  const [view, setView] = useState<"iso" | "close" | "top">("iso");
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);
  const [pv, setPv] = useState<Preview | null>(null);

  const meta = useMeta(gripper);
  const sim = useSimulation(gripper, method, seed);
  const s = sim.session;

  // yöntem listesi değişince geçerli bir seçim kalsın
  useEffect(() => {
    if (methods.length && !methods.some((m) => m.id === method)) setMethod("heuristic");
  }, [methods, method]);

  useEffect(() => {
    if (meta && angle >= meta.angles_deg.length) setAngle(0);
  }, [meta, angle]);

  // otomatik oynatma: animasyon bitince sıradaki adım
  const stepRef = useRef(sim.step);
  stepRef.current = sim.step;
  useEffect(() => {
    if (!playing || !s) return;
    if (s.done) {
      setPlaying(false);
      return;
    }
    if (!sim.busy) {
      const t = window.setTimeout(() => stepRef.current(), 350 / speed);
      return () => window.clearTimeout(t);
    }
  }, [playing, s, sim.busy, speed]);

  // fareyle gezilen eylemin ajan gözünden önizlemesi
  const hoverAction: GraspAction | null = hover ? { angle_idx: angle, ...hover } : null;
  const reqId = useRef(0);
  useEffect(() => {
    if (!s || !hoverAction) {
      setPv(null);
      return;
    }
    const id = ++reqId.current;
    const t = window.setTimeout(() => {
      api.preview(s.id, hoverAction).then((p) => id === reqId.current && setPv(p)).catch(() => undefined);
    }, 60);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s?.id, s?.steps, hover?.row, hover?.col, angle]);

  const history = s?.history ?? [];
  const totals = useMemo(() => {
    const t = { reward: 0, success: 0, collision: 0, fail: 0 };
    for (const h of history) {
      t.reward += h.reward;
      t[h.outcome] += 1;
    }
    return t;
  }, [history]);

  const learns = !!s?.learns;
  const last = sim.lastEvent;
  const footprints: Footprint[] = [];
  if (last && !sim.event) footprints.push({ action: last.action, color: OUTCOME[last.outcome].color });
  if (hoverAction && manual) footprints.push({ action: hoverAction, color: pv?.wall_hit ? C.collision : C.accent, dashed: true });

  const overlayData = overlay === "q" && s?.qmap ? { kind: "q" as const, values: s.qmap.q }
    : overlay === "depth" && s ? { kind: "depth" as const, values: s.depth } : null;

  const methodLabel = methods.find((m) => m.id === method)?.label ?? method;
  const lookingAt = manual && pv ? pv : null;

  if (!meta) return <div className="note">Yükleniyor…</div>;

  return (
    <div>
      <PageHead kicker="Adım 4 · Canlı simülasyon" title="Simülasyon">
        Rastgele üretilmiş bir kutuda ajanı adım adım çalıştır ya da <b>manuel modda</b> kavrama noktasını kendin seç.
        Her kavramadan sonra derinlik kamerası kutuyu yeniden tarar ve ajan yeni haritaya göre karar verir.
      </PageHead>
      {sim.error && <p className="err">Sunucuya ulaşılamadı: {sim.error}. <code>python serve.py</code> çalışıyor mu?</p>}

      <div className="split">
        <div className="stack">
          {s ? (
            <RobotCell meta={meta} parts={sim.parts} depth={s.depth} event={sim.event} eventKey={sim.eventKey}
              onEventDone={sim.onEventDone} speed={speed} overlay={overlayData} height={640} view={view}
              signal={last ? last.outcome : "idle"}
              preview={manual && hoverAction ? { action: hoverAction, color: pv?.wall_hit ? C.collision : C.accent } : null}>
              <div className="overlay-tl hud">
                <span className="dot" style={{ background: sim.event ? C.fail : C.success }} />
                {sim.event ? "Robot çalışıyor" : s.done ? "Bölüm bitti" : "Hazır"} · <b>{methodLabel}</b>
              </div>
              <div className="overlay-tr hud">
                Adım <b>{s.steps}</b> / {s.max_steps} · Kalan <b>{s.parts_left}</b> / {s.n_initial}
              </div>
              <div className="overlay-bl" style={{ pointerEvents: "auto" }}>
                <Seg value={view} onChange={setView} options={[
                  { value: "iso", label: "Hücre" }, { value: "close", label: "Yakın" }, { value: "top", label: "Üstten" },
                ]} />
              </div>
            </RobotCell>
          ) : (
            <div className="viewport" style={{ height: 640 }} />
          )}
          <OutcomeLegend />
        </div>

        <div className="stack">
          <div className="card stack" style={{ gap: 14 }}>
            <Field label="Tutucu">
              <Seg value={gripper} onChange={(g) => { setPlaying(false); setGripper(g); }} disabled={sim.busy}
                options={[{ value: "parallel_jaw", label: "Parmaklı" }, { value: "vacuum", label: "Vakumlu" }]} />
            </Field>
            <Field label="Kararı veren">
              <select value={method} onChange={(e) => { setPlaying(false); setMethod(e.target.value); }} disabled={sim.busy}>
                {methods.map((m) => <option key={m.id} value={m.id}>{m.learns ? `${m.label} (öğrenen)` : m.label}</option>)}
              </select>
            </Field>
            <div className="row" style={{ alignItems: "flex-end" }}>
              <Field label="Kutu tohumu">
                <input type="number" value={seed} min={0} style={{ width: 92 }} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
              </Field>
              <Field label="Animasyon hızı">
                <Seg value={speed} onChange={setSpeed} options={[0.25, 0.5, 1, 2, 4].map((v) => ({ value: v, label: `${v}×` }))} />
              </Field>
            </div>
            <div className="row">
              <button className="btn" disabled={sim.busy} onClick={() => { setPlaying(false); sim.reset(seed); }}>Yeni kutu</button>
              <button className="btn" disabled={sim.busy || !s || s.done || manual} onClick={() => sim.step()}>1 adım</button>
              <button className={`btn ${playing ? "" : "primary"}`} disabled={!s || s.done || manual}
                onClick={() => setPlaying((p) => !p)}>
                {playing ? "Durdur" : "Kutuyu boşalt"}
              </button>
            </div>
            <label className="check">
              <input type="checkbox" checked={manual} onChange={(e) => { setPlaying(false); setManual(e.target.checked); }} />
              Manuel mod: kavrama noktasını ben seçeyim
            </label>
            {manual && (
              <Field label="Bilek açısı θ">
                <Seg value={angle} onChange={setAngle} options={meta.angles_deg.map((d, k) => ({ value: k, label: `${d}°` }))} />
              </Field>
            )}
            <Field label="3D katman">
              <Seg value={overlay} onChange={setOverlay} options={[
                { value: "none", label: "Yok" }, { value: "depth", label: "Derinlik" },
                { value: "q", label: learns ? "Q-haritası" : "Q (eğitim yok)" },
              ]} />
            </Field>
          </div>

          <Banner event={last} />
          <div className="grid g2" style={{ gap: 10 }}>
            <Stat label="Toplam ödül" value={fmt(totals.reward, 1)} color={totals.reward >= 0 ? C.success : C.collision} />
            <Stat label="Başarılı" value={totals.success} color={C.success} />
            <Stat label="Çarpışma" value={totals.collision} color={C.collision} />
            <Stat label="Başarısız" value={totals.fail} color={C.fail} />
          </div>
          <div className="card tight">
            <div className="card-k" style={{ color: C.muted }}>Bölüm akışı</div>
            <Timeline outcomes={history.map((h) => h.outcome)} total={s?.max_steps} />
          </div>
        </div>
      </div>

      <div className="grid g3" style={{ marginTop: 18 }}>
        <div className="card tight">
          <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}>
            <h3>Derinlik kamerası</h3>
            <span className="hint">{manual ? "tıkla: bu noktadan kavra" : "gezin: ajanın gözünden bak"}</span>
          </div>
          {s && (
            <GridCanvas meta={meta} mode="depth" values={s.depth} size={440} footprints={footprints}
              onHover={setHover}
              onClick={manual ? (cell) => sim.step({ angle_idx: angle, ...cell }) : undefined} />
          )}
        </div>
        <div className="card tight">
          <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}>
            <h3>Q-haritası</h3><span className="hint">max<sub>θ</sub> Q</span>
          </div>
          {s?.qmap ? (
            <GridCanvas meta={meta} mode="q" values={s.qmap.q} size={440} footprints={footprints} onHover={setHover}
              onClick={manual ? (cell) => sim.step({ angle_idx: angle, ...cell }) : undefined}
              tooltip={(r, c) => {
                const q = s.qmap!.q[r]?.[c];
                return q == null ? null : `Q ${fmt(q, 3)} · θ* ${meta.angles_deg[s.qmap!.best_angle[r][c]]}°`;
              }} />
          ) : (
            <Empty title="Bu yöntem öğrenmiyor">
              <p className="muted">Q-haritası yalnızca eğitilmiş Q-learning / SARSA ajanları için vardır. <a href="#/egitim">Eğitim</a> sayfasından bir ajan eğit.</p>
            </Empty>
          )}
        </div>
        <Card kicker={lookingAt ? "Fare altındaki eylem" : last ? `Son karar · adım ${last.step}` : "Ajanın gözünden"}
          title={lookingAt ? `(${hover?.col}, ${hover?.row}) · θ = ${meta.angles_deg[angle]}°` : last ? `(${last.action.col}, ${last.action.row}) · θ = ${Math.round(last.action.theta_deg ?? 0)}°` : "Henüz karar yok"}>
          {(lookingAt?.features ?? last?.features) && <FeatureChips features={(lookingAt?.features ?? last?.features)!} />}
          {lookingAt && (
            <p style={{ marginTop: 8 }}>
              Tablo hücresi <b className="mono">#{lookingAt.state_id}</b>
              {lookingAt.q != null && <> · Q = <b className="mono">{fmt(lookingAt.q, 3)}</b></>}
              {lookingAt.wall_hit && <span style={{ color: C.collision }}> · duvara çarpar</span>}
            </p>
          )}
          {!lookingAt && last && (
            <p style={{ marginTop: 8 }}>
              {last.chosen_by === "user" ? "Bu noktayı sen seçtin." : "Bu noktayı ajan seçti."}
              {last.q != null && <> Tablodaki değeri Q = <b className="mono">{fmt(last.q, 3)}</b>.</>}
            </p>
          )}
          {!lookingAt && !last && <p>Haritanın üzerinde gezinince ajanın o eylem için gördüğü özellikler burada çıkar.</p>}
        </Card>
      </div>

      {history.length > 0 && (
        <div className="card flush" style={{ marginTop: 18 }}>
          <div className="card-head"><h3>Adım günlüğü</h3><span className="hint">{history.length} kavrama</span></div>
          <div style={{ maxHeight: 340, overflowY: "auto" }}>
            <table className="t">
              <thead><tr><th>#</th><th>Kim</th><th>Nokta</th><th>θ</th><th>Sonuç</th><th>Açıklama</th><th className="num">Q</th><th className="num">Ödül</th></tr></thead>
              <tbody>
                {[...history].reverse().map((h) => (
                  <tr key={h.step}>
                    <td className="mono">{h.step}</td>
                    <td className="muted">{h.chosen_by === "user" ? "sen" : "ajan"}</td>
                    <td className="mono">({h.action.col}, {h.action.row})</td>
                    <td className="mono">{Math.round(h.action.theta_deg ?? 0)}°</td>
                    <td style={{ color: OUTCOME[h.outcome].color, fontWeight: 700 }}>{OUTCOME[h.outcome].label}</td>
                    <td className="muted">{h.reason}</td>
                    <td className="num">{h.q == null ? "—" : fmt(h.q, 3)}</td>
                    <td className="num" style={{ color: OUTCOME[h.outcome].color }}>{h.reward > 0 ? "+" : ""}{h.reward.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
