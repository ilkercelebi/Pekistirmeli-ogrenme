import { useEffect, useState } from "react";
import { api, type EvalResult, type GripperKind } from "../api";
import { GroupedBars, METHOD_COLOR, SERIES } from "../components/Charts";
import { Card, Field, PageHead, Section, Seg } from "../components/ui";
import { useMethods } from "../hooks";
import { C, fmt, pct } from "../theme";

const METRICS: { key: keyof EvalResult; label: string; good: "high" | "low" }[] = [
  { key: "success_rate", label: "Başarı oranı", good: "high" },
  { key: "collision_rate", label: "Çarpışma oranı", good: "low" },
  { key: "fail_rate", label: "Başarısız kavrama", good: "low" },
  { key: "cleared", label: "Boşaltılan parça", good: "high" },
];

export default function Compare() {
  const [gripper, setGripper] = useState<GripperKind>("parallel_jaw");
  const methods = useMethods(gripper);
  const [selected, setSelected] = useState<string[]>(["random", "heuristic"]);
  const [episodes, setEpisodes] = useState(30);
  const [results, setResults] = useState<EvalResult[] | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eğitilmiş ajanlar varsa varsayılan olarak seçili gelsin
    setSelected(["random", "heuristic", ...methods.filter((m) => m.learns).map((m) => m.id)]);
    setResults(null);
  }, [methods]);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await api.evaluate(gripper, selected, episodes);
      setResults(r.results);
    } catch (e) {
      setError(String((e as Error).message));
    }
    setRunning(false);
  };

  const colorOf = (id: string, i: number) => METHOD_COLOR[id] ?? SERIES[i % SERIES.length];
  const chart = METRICS.map((m) => {
    const row: Record<string, number | string> = { metric: m.label };
    results?.forEach((r) => (row[r.method] = r[m.key] as number));
    return row;
  });

  const best = results && results.length > 1
    ? [...results].sort((a, b) => b.success_rate - a.success_rate)[0]
    : null;
  const heur = results?.find((r) => r.method === "heuristic");

  return (
    <div>
      <PageHead kicker="Adım 5 · Değerlendirme" title="Karşılaştırma">
        Bütün yöntemler aynı test kutularında, keşif kapalıyken çalışır. Test kutuları eğitimde hiç görülmemiş sabit
        tohumlardan üretilir; böylece karşılaştırma adil olur.
      </PageHead>

      <div className="card">
        <div className="grid g3" style={{ alignItems: "end" }}>
          <Field label="Tutucu">
            <Seg value={gripper} onChange={setGripper} options={[{ value: "parallel_jaw", label: "Parmaklı" }, { value: "vacuum", label: "Vakumlu" }]} />
          </Field>
          <Field label="Test kutusu sayısı">
            <Seg value={episodes} onChange={setEpisodes} options={[20, 30, 60, 100].map((v) => ({ value: v, label: String(v) }))} />
          </Field>
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <button className="btn primary" onClick={run} disabled={running || selected.length === 0}>
              {running ? "Çalışıyor…" : "Değerlendir"}
            </button>
          </div>
        </div>
        <div className="row" style={{ marginTop: 14 }}>
          {methods.map((m, i) => (
            <label key={m.id} className="check" style={{ marginRight: 14 }}>
              <input type="checkbox" checked={selected.includes(m.id)}
                onChange={(e) => setSelected((s) => (e.target.checked ? [...s, m.id] : s.filter((x) => x !== m.id)))} />
              <span style={{ color: colorOf(m.id, i) }}>●</span> {m.label}{m.learns ? " (öğrenen)" : ""}
            </label>
          ))}
        </div>
        {methods.every((m) => !m.learns) && (
          <p className="note" style={{ marginTop: 10 }}>
            Bu tutucu için eğitilmiş ajan yok; şimdilik yalnızca rastgele ve sezgisel yöntemler karşılaştırılabilir.{" "}
            <a href="#/egitim">Eğitim sayfası</a>.
          </p>
        )}
        {error && <p className="err" style={{ marginTop: 10 }}>{error}</p>}
        {running && <p className="note" style={{ marginTop: 10 }}>Her yöntem {episodes} kutuyu boşaltmaya çalışıyor; bu birkaç saniye sürebilir.</p>}
      </div>

      {results && (
        <div className="fade-in">
          <Section num="1" title="Sonuçlar">{results[0] && `${episodes} test kutusu üzerinde ortalama.`}</Section>
          <div className="card">
            <GroupedBars data={chart} groups={results.map((r, i) => ({ key: r.method, name: r.label, color: colorOf(r.method, i) }))} />
          </div>

          <div className="card flush" style={{ marginTop: 16 }}>
            <table className="t">
              <thead>
                <tr>
                  <th>Yöntem</th>
                  {METRICS.map((m) => <th key={m.key} className="num">{m.label}</th>)}
                  <th className="num">Getiri</th><th className="num">Adım</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={r.method}>
                    <td style={{ fontWeight: 650 }}><span style={{ color: colorOf(r.method, i) }}>●</span> {r.label}</td>
                    {METRICS.map((m) => {
                      const vals = results.map((x) => x[m.key] as number);
                      const top = m.good === "high" ? Math.max(...vals) : Math.min(...vals);
                      const v = r[m.key] as number;
                      return (
                        <td key={m.key} className="num" style={{ color: v === top && results.length > 1 ? C.success : undefined }}>
                          {pct(v)} <span className="note">± {Math.round((r[`${m.key}_std`] as number) * 100)}</span>
                        </td>
                      );
                    })}
                    <td className="num">{fmt(r.return, 2)}</td>
                    <td className="num">{fmt(r.steps, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid g2" style={{ marginTop: 16 }}>
            {best && (
              <Card kicker="Bulgu" title={`En yüksek başarı oranı: ${best.label}`} tone={C.success}>
                Kavramaların <b>{pct(best.success_rate)}</b> kadarı başarılı, ortalama <b>{fmt(best.steps, 1)}</b> adımda kutunun{" "}
                <b>{pct(best.cleared)}</b> kadarı boşaltıldı.
                {heur && best.method !== "heuristic" && (
                  <> Sezgisel kurala göre fark <b>{Math.round((best.success_rate - heur.success_rate) * 100)} puan</b>.</>
                )}
              </Card>
            )}
            <Card title="Ölçütler nasıl okunur?">
              <b>Başarı oranı</b> = başarılı kavrama / toplam deneme. <b>Boşaltılan parça</b> = kutudan alınan parça / başlangıçtaki
              parça. Çarpışma ve başarısız kavrama düşük olmalı. ± değerleri kutular arası standart sapmadır.
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
