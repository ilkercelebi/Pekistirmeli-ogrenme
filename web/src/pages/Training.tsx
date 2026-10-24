import { useCallback, useEffect, useState } from "react";
import { api, type GripperKind, type Job, type RunDetail, type RunSummary } from "../api";
import { LineSeries, SignedBars } from "../components/Charts";
import { Card, Chip, Empty, Field, PageHead, Section, Seg, Stat } from "../components/ui";
import { C, FEATURE_TITLE, fmt, pct } from "../theme";
import { useMeta } from "../hooks";

function TrainForm({ onFinished }: { onFinished: () => void }) {
  const [algo, setAlgo] = useState("q_learning");
  const [gripper, setGripper] = useState<GripperKind>("parallel_jaw");
  const [episodes, setEpisodes] = useState(1500);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.currentJob().then((j) => j && setJob(j)).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = window.setInterval(async () => {
      try {
        const j = await api.job(job.id);
        setJob(j);
        if (j.status !== "running") onFinished();
      } catch {
        /* geçici hata */
      }
    }, 800);
    return () => window.clearInterval(t);
  }, [job, onFinished]);

  const start = async () => {
    setError(null);
    try {
      setJob(await api.startTrain(algo, gripper, episodes));
    } catch (e) {
      setError(String((e as Error).message));
    }
  };

  const running = job?.status === "running";
  const progress = job ? job.episode / job.episodes : 0;
  const eta = job && job.episode > 0 && running ? (job.elapsed / job.episode) * (job.episodes - job.episode) : 0;
  const lastEval = job?.evals[job.evals.length - 1];

  return (
    <div className="split">
      <div className="card stack" style={{ gap: 14 }}>
        <Field label="Algoritma">
          <Seg value={algo} onChange={setAlgo} disabled={running}
            options={[{ value: "q_learning", label: "Q-learning" }, { value: "sarsa", label: "SARSA" }, { value: "double_q", label: "Double Q" }]} />
        </Field>
        <Field label="Tutucu">
          <Seg value={gripper} onChange={setGripper} disabled={running}
            options={[{ value: "parallel_jaw", label: "Parmaklı" }, { value: "vacuum", label: "Vakumlu" }]} />
        </Field>
        <Field label="Bölüm sayısı">
          <Seg value={episodes} onChange={setEpisodes} disabled={running}
            options={[300, 1500, 3000, 5000].map((v) => ({ value: v, label: v.toLocaleString("tr-TR") }))} />
        </Field>
        <p className="note">
          Bir bölüm bir kutuyu boşaltma denemesidir. ε, eğitimin ilk yarısında 1.0'dan 0.05'e iner. 1.500 bölüm yaklaşık
          4–5 dakika sürer. Sonuç <code>results/</code> klasörüne kaydedilir ve diğer sayfalarda seçilebilir.
        </p>
        <div className="row">
          <button className="btn primary" onClick={start} disabled={running}>Eğitimi başlat</button>
          {running && <button className="btn" onClick={() => api.cancelJob(job!.id)}>Durdur</button>}
        </div>
        {error && <p className="err">{error}</p>}
      </div>

      <div className="card">
        {!job ? (
          <Empty title="Çalışan eğitim yok">
            <p className="muted">Eğitimi başlatınca öğrenme eğrisi burada canlı çizilir.</p>
          </Empty>
        ) : (
          <div className="stack">
            <div className="row">
              <h3 style={{ margin: 0 }}>{job.name}</h3>
              <span className="spacer" />
              <Chip color={running ? C.fail : job.status === "done" ? C.success : C.muted}>
                {running ? "eğitiliyor" : job.status === "done" ? "tamamlandı" : job.status === "cancelled" ? "durduruldu" : "hata"}
              </Chip>
            </div>
            <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>
            <div className="grid g2" style={{ gap: 10 }}>
              <Stat label="Bölüm" value={`${job.episode}`} sub={`/ ${job.episodes}`} />
              <Stat label="Süre" value={`${Math.round(job.elapsed)} sn`} sub={running ? `~${Math.round(eta)} sn kaldı` : ""} />
              <Stat label="Test başarısı" value={pct(lastEval?.success_rate)} color={C.success} sub="açgözlü politika" />
              <Stat label="Test çarpışma" value={pct(lastEval?.collision_rate)} color={C.collision} />
            </div>
            <LineSeries data={job.curve as unknown as Record<string, number>[]} x="episode" height={260} yDomain={[0, 1]} percent
              lines={[
                { key: "success_rate", name: "Başarı oranı", color: C.success },
                { key: "collision_rate", name: "Çarpışma oranı", color: C.collision },
                { key: "epsilon", name: "ε (keşif)", color: C.muted, dashed: true },
              ]} />
            {job.error && <p className="err">{job.error}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

function RunView({ name }: { name: string }) {
  const [d, setD] = useState<RunDetail | null>(null);
  const [metric, setMetric] = useState<"success_rate" | "collision_rate" | "cleared" | "return">("success_rate");
  const meta = useMeta("parallel_jaw");
  useEffect(() => {
    setD(null);
    api.run(name, 50).then(setD).catch(() => setD(null));
  }, [name]);
  if (!d) return <div className="card note">Yükleniyor…</div>;
  const labels: Record<string, string> = { success_rate: "Başarı oranı", collision_rate: "Çarpışma oranı", cleared: "Boşaltılan parça", return: "Bölüm getirisi" };
  const featureBins = (n: string) => meta?.features.find((f) => f.name === n)?.bins.map((b) => b.label) ?? d.marginal[n].map((_, i) => String(i));

  return (
    <div className="stack fade-in">
      <div className="grid g2">
        <div className="card">
          <div className="row" style={{ marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Öğrenme eğrisi</h3>
            <span className="spacer" />
            <Seg value={metric} onChange={setMetric} options={Object.entries(labels).map(([k, v]) => ({ value: k as typeof metric, label: v }))} />
          </div>
          <LineSeries data={d.curve as unknown as Record<string, number>[]} x="episode" height={280}
            percent={metric !== "return"} yDomain={metric === "return" ? undefined : [0, 1]}
            lines={[{ key: metric, name: `${labels[metric]} (50 bölüm ort.)`, color: C.accent }, ...(metric !== "return" ? [{ key: "epsilon", name: "ε", color: C.muted, dashed: true }] : [])]} />
          <p className="note">Eğitim sırasında keşif açık olduğu için bu eğri gerçek başarımı olduğundan düşük gösterir; sağdaki eğri keşif kapalı (ε = 0) ölçümdür.</p>
        </div>
        <div className="card">
          <h3>Açgözlü politika (test kutuları)</h3>
          <LineSeries data={d.evals as unknown as Record<string, number>[]} x="episode" height={280} yDomain={[0, 1]} percent
            lines={[
              { key: "success_rate", name: "Başarı", color: C.success, dot: true },
              { key: "collision_rate", name: "Çarpışma", color: C.collision, dot: true },
              { key: "cleared", name: "Boşaltma", color: C.info, dot: true },
            ]} />
        </div>
      </div>

      <Section num="3" title="Ajan ne öğrendi?">
        Her özelliğin her kovası için, o kovaya düşen tüm tablo hücrelerinin ziyaret ağırlıklı ortalama Q değeri. Yeşil:
        ajan bu durumu iyi buluyor, kırmızı: kaçınıyor. Tablo yöntemi bu sayede yorumlanabilir.
        {` Tablonun ${d.visited} / ${d.n_states} hücresi ziyaret edildi.`}
      </Section>
      <div className="grid g4">
        {Object.entries(d.marginal).map(([n, vals]) => (
          <div key={n} className="card tight">
            <div className="card-k" style={{ color: C.muted }}>{FEATURE_TITLE[n] ?? n}</div>
            <SignedBars data={vals.map((v, i) => ({ label: featureBins(n)[i], value: v }))} />
          </div>
        ))}
      </div>

      <div className="grid g2">
        {[["En yüksek Q değerli durumlar", d.best, C.success], ["En düşük Q değerli durumlar", d.worst, C.collision]].map(([title, rows, color]) => (
          <div key={title as string} className="card flush">
            <div className="card-head"><h3>{title as string}</h3><span className="hint">en az 10 ziyaret</span></div>
            <table className="t">
              <tbody>
                {(rows as RunDetail["best"]).map((r) => (
                  <tr key={r.state_id}>
                    <td className="num" style={{ color: color as string, width: 70 }}>{fmt(r.q, 2)}</td>
                    <td>{r.features.map((f) => <Chip key={f.name} tone={f.tone}>{f.label}</Chip>)}</td>
                    <td className="num note">{r.visits}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Training() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const load = useCallback(() => {
    api.runs().then((r) => {
      setRuns(r);
      setSel((s) => s ?? r[r.length - 1]?.name ?? null);
    }).catch(() => setRuns([]));
  }, []);
  useEffect(load, [load]);

  return (
    <div>
      <PageHead kicker="Adım 6 · Öğrenme" title="Eğitim ve sonuçlar">
        Ajan, binlerce rastgele kutuda deneme-yanılma yaparak Q tablosunu doldurur. Buradan yeni bir eğitim başlatabilir,
        biten eğitimlerin öğrenme eğrilerini ve tablonun ne öğrendiğini inceleyebilirsin.
      </PageHead>

      <Section num="1" title="Yeni eğitim" />
      <TrainForm onFinished={load} />

      <Section num="2" title="Kayıtlı eğitimler">
        Komut satırından (<code>python train.py</code>) ya da bu sayfadan yapılan eğitimler.
      </Section>
      {runs.length === 0 ? (
        <div className="card"><Empty title="Henüz eğitim yok">Yukarıdan bir eğitim başlat.</Empty></div>
      ) : (
        <div className="stack">
          <div className="grid g4">
            {runs.map((r) => (
              <button key={r.name} className="card tight" onClick={() => setSel(r.name)}
                style={{ textAlign: "left", cursor: "pointer", borderColor: sel === r.name ? C.accent : undefined }}>
                <div className="card-k" style={{ color: sel === r.name ? C.accent : C.muted }}>
                  {r.gripper === "vacuum" ? "Vakumlu" : "Parmaklı"}
                </div>
                <h3>{r.label}</h3>
                <p>{r.episodes.toLocaleString("tr-TR")} bölüm · test başarısı <b>{pct(r.last_eval?.success_rate)}</b></p>
              </button>
            ))}
          </div>
          {sel && <RunView name={sel} />}
        </div>
      )}
      {runs.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <Card title="Sonraki adım">
            Eğitilmiş ajanı <a href="#/simulasyon">Simülasyon</a> sayfasında "Kararı veren" listesinden seçerek izleyebilir,{" "}
            <a href="#/karsilastirma">Karşılaştırma</a> sayfasında diğer yöntemlerle karşılaştırabilirsin.
          </Card>
        </div>
      )}
    </div>
  );
}
