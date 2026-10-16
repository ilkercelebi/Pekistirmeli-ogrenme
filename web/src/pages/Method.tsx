import { useEffect, useState } from "react";
import { api, type Preview } from "../api";
import GridCanvas from "../components/GridCanvas";
import { Card, Chip, Empty, FeatureChips, PageHead, Section, Seg, Stat, TeX } from "../components/ui";
import { useMeta, useMethods, useSimulation } from "../hooks";
import { C, FEATURE_TITLE, TONE, fmt } from "../theme";

function QMapSection() {
  const meta = useMeta("parallel_jaw");
  const methods = useMethods("parallel_jaw").filter((m) => m.learns);
  const [run, setRun] = useState<string | null>(null);
  useEffect(() => {
    if (!run && methods.length) setRun(methods[methods.length - 1].id);
  }, [methods, run]);
  if (!methods.length || !run || !meta) {
    return (
      <div className="card">
        <Empty title="Henüz eğitilmiş ajan yok">
          <p className="muted" style={{ marginBottom: 14 }}>Q-haritası, öğrenilmiş bir Q tablosundan üretilir.</p>
          <a className="btn primary" href="#/egitim">Eğitim sayfasına git</a>
        </Empty>
      </div>
    );
  }
  return <QMapView run={run} runs={methods.map((m) => m.id)} onRun={setRun} />;
}

function QMapView({ run, runs, onRun }: { run: string; runs: string[]; onRun: (r: string) => void }) {
  const meta = useMeta("parallel_jaw");
  const sim = useSimulation("parallel_jaw", run, 11, false);
  const s = sim.session;
  if (!meta || !s?.qmap) return <div className="card note">Yükleniyor…</div>;
  return (
    <div className="stack">
      <div className="row">
        <select value={run} onChange={(e) => onRun(e.target.value)}>
          {runs.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <button className="btn sm" onClick={() => sim.reset((s.seed ?? 0) + 1)}>Başka kutu</button>
      </div>
      <div className="grid g2">
        <div className="card tight">
          <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}><h3>Yükseklik haritası</h3></div>
          <GridCanvas meta={meta} mode="depth" values={s.depth} size={460} />
        </div>
        <div className="card tight">
          <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}>
            <h3>Q-haritası</h3><span className="hint">her pikselde max<sub>θ</sub> Q</span>
          </div>
          <GridCanvas meta={meta} mode="q" values={s.qmap.q} size={460}
            tooltip={(r, c) => {
              const q = s.qmap!.q[r]?.[c];
              return q == null ? null : `Q = ${fmt(q, 3)} · en iyi θ = ${meta.angles_deg[s.qmap!.best_angle[r][c]]}°`;
            }} />
        </div>
      </div>
      <p className="note">
        Yeşil bölgeler ajanın iyi bulduğu kavrama noktaları, kırmızı bölgeler duvar kenarları ve ajanın kaçındığı noktalar.
        Bu harita bir sinir ağının çıktısı değil: her piksel, tablodaki bir hücrenin değeri.
      </p>
    </div>
  );
}

export default function Method() {
  const meta = useMeta("parallel_jaw");
  const sim = useSimulation("parallel_jaw", "heuristic", 21, false);
  const s = sim.session;
  const [angle, setAngle] = useState(2);
  const [maps, setMaps] = useState<Record<string, number[][]> | null>(null);
  const [pick, setPick] = useState<{ row: number; col: number } | null>(null);
  const [pv, setPv] = useState<Preview | null>(null);

  useEffect(() => {
    if (s) api.featureMaps(s.id, angle).then(setMaps).catch(() => setMaps(null));
  }, [s, angle]);

  useEffect(() => {
    if (!s) return;
    const cell = pick ?? (() => {
      const p = s.parts.filter((q) => q.alive).sort((a, b) => b.base + b.thickness - a.base - a.thickness)[0];
      return p ? { row: Math.round(p.cy), col: Math.round(p.cx) } : { row: 22, col: 22 };
    })();
    if (!pick) setPick(cell);
    api.preview(s.id, { angle_idx: angle, ...cell }).then(setPv).catch(() => setPv(null));
  }, [s, pick, angle]);

  const dims = meta?.dims ?? [];
  const bins = pv?.features.map((f) => f.bin ?? 0) ?? [];
  const terms = bins.map((b, i) => `${b}\\cdot${dims.slice(i + 1).reduce((a, d) => a * d, 1)}`).join(" + ");

  return (
    <div>
      <PageHead kicker="Adım 3 · Tablo tabanlı pekiştirmeli öğrenme" title="Yöntem">
        Ödevde önerilen yöntem, görüntüden piksel bazlı Q-haritası üreten bir derin ağdır (FCN-DQN). Biz bilinçli olarak
        <b> klasik pekiştirmeli öğrenme</b> kullandık: Q değerleri bir sinir ağında değil, birkaç yüz hücrelik okunabilir
        bir <b>tabloda</b> tutulur.
      </PageHead>

      <div className="grid g4">
        <Stat label="Ham durum" value="∞" sub="44×44 sürekli yükseklik haritası" />
        <Stat label="Aday eylem / adım" value={meta ? meta.candidates.toLocaleString("tr-TR") : "—"} sub="nokta × açı" />
        <Stat label="Tablo hücresi" value={meta ? meta.n_states.toLocaleString("tr-TR") : "—"} sub={dims.join(" × ")} color={C.accent} />
        <Stat label="Öğrenilen parametre" value={meta ? meta.n_states.toLocaleString("tr-TR") : "—"} sub="sadece Q değerleri" color={C.success} />
      </div>

      <Section num="1" title="Fikir: her eylemi birkaç ayrık özelliğe indirgemek">
        Tüm yükseklik haritasını tablo satırı yapmak imkânsızdır. Bunun yerine her aday eylem a = (x, y, θ) için o noktanın
        çevresinden aşağıdaki ölçümler alınır ve kovalara ayrılır. Ölçümler tutucu geometrisinden türetilen sabit
        filtrelerle hesaplanır; neyin iyi neyin kötü olduğunu ajan ödülden öğrenir.
      </Section>
      <div className="card flush">
        <table className="t">
          <thead><tr><th>Özellik</th><th>Kovalar</th><th>Ne ölçüyor?</th></tr></thead>
          <tbody>
            {meta?.features.map((f) => (
              <tr key={f.name}>
                <td style={{ fontWeight: 650, whiteSpace: "nowrap" }}>{FEATURE_TITLE[f.name]}</td>
                <td>{f.bins.map((b) => <Chip key={b.label} tone={b.tone}>{b.label}</Chip>)}</td>
                <td className="muted">{f.info}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Section num="2" title="Özellik haritaları">
        Aynı kutu için her özelliğin tüm piksellerdeki değeri. Açıya bağlı özellikler (duvar, parmak boşluğu, genişlik) θ
        değiştikçe değişir. Bir piksele tıklayınca o eylemin tablo indeksi aşağıda hesaplanır.
      </Section>
      {meta && s && (
        <div className="stack">
          <div className="row">
            <Seg value={angle} onChange={setAngle} options={meta.angles_deg.map((d, k) => ({ value: k, label: `θ = ${d}°` }))} />
            <button className="btn sm" onClick={() => { setPick(null); sim.reset(s.seed + 1); }}>Başka kutu</button>
          </div>
          <div className="grid g3">
            <div className="card tight">
              <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}><h3>Yükseklik haritası</h3><span className="hint">girdi</span></div>
              <GridCanvas meta={meta} mode="depth" values={s.depth} size={320} onClick={setPick}
                footprints={pick ? [{ action: { angle_idx: angle, ...pick }, color: C.accent }] : []} />
            </div>
            {maps && meta.features.filter((f) => maps[f.name]).map((f) => (
              <div key={f.name} className="card tight">
                <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}>
                  <h3>{FEATURE_TITLE[f.name]}</h3>
                  <span className="hint">{f.bins.map((b) => b.label).join(" / ")}</span>
                </div>
                <GridCanvas meta={meta} mode="discrete" values={maps[f.name]} size={320} onClick={setPick}
                  discreteColors={f.bins.map((b) => TONE[b.tone])}
                  footprints={pick ? [{ action: { angle_idx: angle, ...pick }, color: "#ffffff" }] : []}
                  tooltip={(r, c) => {
                    const v = maps[f.name][r]?.[c];
                    return v == null || v < 0 ? null : f.bins[v]?.label ?? null;
                  }} />
              </div>
            ))}
          </div>
        </div>
      )}

      <Section num="3" title="Özelliklerden tablo indeksine">
        Seçilen eylemin özellik kovaları tek bir tamsayıya kodlanır (karışık tabanlı sayı). Bu sayı Q tablosunun satır numarasıdır.
      </Section>
      {pv && pick && (
        <div className="grid g2">
          <Card kicker={`Eylem (${pick.col}, ${pick.row}) · θ = ${meta?.angles_deg[angle]}°`} title="Ajanın gözünden bu eylem">
            <FeatureChips features={pv.features} />
            {pv.wall_hit && <p className="err" style={{ marginTop: 6 }}>Bu eylem gerçekte duvara çarpar.</p>}
          </Card>
          <div className="stack">
            <TeX>{"\\phi(s,a) = \\sum_i b_i \\prod_{j>i} d_j"}</TeX>
            <TeX>{`\\phi = ${terms} = \\mathbf{${pv.state_id}}`}</TeX>
            <p className="note">
              b<sub>i</sub>: i. özelliğin kovası, d<sub>j</sub>: j. özelliğin kova sayısı. Q(s, a) ≈ Q<sub>tablo</sub>[φ(s, a)].
            </p>
          </div>
        </div>
      )}

      <Section num="4" title="Öğrenme kuralı">
        Ajan her kavramadan sonra yalnızca o eylemin tablo hücresini günceller. Hedef, alınan ödül artı bir sonraki durumdan
        beklenen indirimli değerdir (zamansal fark öğrenmesi).
      </Section>
      <div className="grid g2">
        <div className="stack">
          <div className="card-k" style={{ color: C.accent }}>Q-learning (off-policy)</div>
          <TeX>{"Q[\\phi(s,a)] \\leftarrow Q[\\phi(s,a)] + \\alpha\\Big(r + \\gamma \\max_{a'} Q[\\phi(s',a')] - Q[\\phi(s,a)]\\Big)"}</TeX>
          <div className="card-k" style={{ color: C.info, marginTop: 8 }}>SARSA (on-policy)</div>
          <TeX>{"Q[\\phi(s,a)] \\leftarrow Q[\\phi(s,a)] + \\alpha\\Big(r + \\gamma\\, Q[\\phi(s',a')] - Q[\\phi(s,a)]\\Big)"}</TeX>
        </div>
        <div className="grid g2">
          <Card title="Keşif: ε-açgözlü">
            ε olasılıkla rastgele eylem, aksi hâlde en yüksek Q. ε, <b>1.0</b>'dan <b>0.05</b>'e doğrusal azalır.
          </Card>
          <Card title="Öğrenme hızı α">
            Ziyaret sayısına bağlı: <b>α = max(0.01, N<sup>−0.6</sup>)</b>. Sık görülen durumlar kararlı hâle gelir.
          </Card>
          <Card title="İndirim γ = 0.5">
            Sonraki parçaları açığa çıkaran alımların değeri de hesaba katılır.
          </Card>
          <Card title="Başlangıç Q = 0">
            Ödüllerin çoğu negatif olduğu için denenmemiş hücreler cazip görünür; bu doğal bir keşif sağlar.
          </Card>
        </div>
      </div>

      <Section num="5" title="Sonuç: tablo değerlerinden Q-haritası">
        Her pikselde, tüm açılar için tablo değerinin en büyüğü alınınca ödevdeki "piksel bazlı Q-haritası" derin öğrenme
        olmadan elde edilir. Ajan her adımda bu haritanın en yeşil noktasını seçer.
      </Section>
      <QMapSection />
    </div>
  );
}
