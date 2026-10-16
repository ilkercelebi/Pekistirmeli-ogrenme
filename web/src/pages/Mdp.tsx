import { useMemo, useState } from "react";
import { Card, PageHead, Section, Stat, TeX } from "../components/ui";
import GridCanvas from "../components/GridCanvas";
import { useMeta, useSimulation } from "../hooks";
import RobotCell from "../three/Scene";
import { C, fmt } from "../theme";

export default function Mdp() {
  const meta = useMeta("parallel_jaw");
  const [seed, setSeed] = useState(7);
  const sim = useSimulation("parallel_jaw", "heuristic", seed, false);
  const s = sim.session;

  // eylem örneği için en yüksek parçanın merkezi
  const focus = useMemo(() => {
    if (!s) return null;
    const alive = s.parts.filter((p) => p.alive);
    if (!alive.length) return null;
    const top = alive.reduce((a, b) => (a.base + a.thickness > b.base + b.thickness ? a : b));
    return { row: Math.round(top.cy), col: Math.round(top.cx) };
  }, [s]);

  return (
    <div>
      <PageHead kicker="Adım 2 · Markov karar süreci" title="MDP tasarımı">
        Pekiştirmeli öğrenmede problem bir Markov karar süreci olarak tanımlanır: ajan bir <b>durum</b> görür, bir
        <b> eylem</b> seçer, ortam bir <b>ödül</b> verir ve yeni duruma geçer. Bu sayfada bin picking için bu üç parçanın
        nasıl tanımlandığını görüyorsun.
      </PageHead>

      <div className="grid g3">
        <Card kicker="Durum  s" title="Yükseklik haritası + tutucu geometrisi" tone={C.info}>
          Kutunun üstündeki RGB-D kamera her pikselde yüksekliği ölçer: <b>44 × 44</b> hücre, 1 hücre = 1 cm.
          Ajan parçaların ne olduğunu ya da nerede olduğunu bilmez; yalnızca bu haritayı (±0.05 cm gürültüyle) görür.
        </Card>
        <Card kicker="Eylem  a" title="Kavrama noktası ve bilek açısı" tone={C.accent}>
          <TeX block={false}>{"a = (x,\\ y,\\ \\theta)"}</TeX> — görüntü koordinatlarında nokta ve
          <b> θ ∈ {"{0°, 45°, 90°, 135°}"}</b>. Her adımda <b>{meta ? meta.candidates.toLocaleString("tr-TR") : "6.400"}</b> aday eylem vardır.
        </Card>
        <Card kicker="Ödül  r" title="Ödev tanımındaki ödül fonksiyonu" tone={C.success}>
          <b style={{ color: C.success }}>+1</b> parça çarpmadan ve düşmeden kalktı ·{" "}
          <b style={{ color: C.collision }}>−1</b> tutucu kutuya çarptı ·{" "}
          <b style={{ color: C.fail }}>−0.2</b> kavrama başarısız.
        </Card>
      </div>

      <Section num="1" title="Durum: kamera ne görüyor?">
        Solda gerçek sahne, sağda ajanın tek girdisi olan yükseklik haritası. Renk yüksekliği gösterir: koyu zemin, açık renk
        yığının tepesi. Haritanın üzerinde gezinince ölçülen yükseklik görünür.
      </Section>
      {meta && s && (
        <div className="split">
          <RobotCell meta={meta} parts={sim.parts} depth={s.depth} height={470} view="iso">
            <div className="overlay-tl hud">Gerçek sahne · kutu #{s.seed}</div>
          </RobotCell>
          <div className="stack">
            <div className="card tight">
              <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}>
                <h3>Derinlik kamerası görüntüsü</h3>
                <span className="hint">ajanın gördüğü</span>
              </div>
              <GridCanvas meta={meta} mode="depth" values={s.depth} size={420}
                tooltip={(r, c) => {
                  const v = s.depth[r]?.[c];
                  const wall = r < meta.grid.wall_thickness || c < meta.grid.wall_thickness ||
                    r >= meta.grid.size - meta.grid.wall_thickness || c >= meta.grid.size - meta.grid.wall_thickness;
                  return wall ? "kutu duvarı" : `(${c}, ${r})  ${fmt(v, 1)} cm`;
                }} />
            </div>
            <div className="row">
              <button className="btn" onClick={() => { const n = seed + 1; setSeed(n); sim.reset(n); }}>Başka kutu</button>
              <span className="note">{s.n_initial} parça · tohum {s.seed}</span>
            </div>
          </div>
        </div>
      )}

      <Section num="2" title="Eylem: nokta ve bilek açısı">
        Aynı noktada dört farklı bilek açısı. Turuncu dikdörtgenler parmakların açık hâlde ineceği yerler, kesikli çizgi
        kapanma yönü. Uzun bir parçayı kısa kenarından sıkıştırmak kolaydır; uzun ekseni boyunca açılan parmaklar ise
        parçanın üstüne oturur.
      </Section>
      {meta && s && focus && (
        <div className="grid g4">
          {meta.angles_deg.map((deg, k) => (
            <div key={deg} className="card tight">
              <div className="card-head" style={{ padding: "0 0 10px", border: 0 }}>
                <h3 className="mono">θ = {deg}°</h3>
              </div>
              <GridCanvas meta={meta} mode="depth" values={s.depth} size={300}
                crop={{ row: focus.row, col: focus.col, radius: 10 }}
                footprints={[{ action: { angle_idx: k, row: focus.row, col: focus.col, theta_deg: deg }, color: C.accent }]} />
            </div>
          ))}
        </div>
      )}

      <Section num="3" title="Ödül: neyi teşvik ediyoruz?">
        Ödül, ajana ne istediğimizi söyleyen tek sinyaldir. Çarpışma cezası başarısızlık cezasından beş kat büyük: ajan
        önce duvardan uzak durmayı, sonra parçayı doğru tutmayı öğrenir.
      </Section>
      <div className="grid g3">
        <Stat label="Başarılı kaldırma" value="+1" color={C.success} sub="kutuya çarpmadan, düşürmeden" />
        <Stat label="Kutuya çarpma" value="−1" color={C.collision} sub="tutucu ayak izi duvara değdi" />
        <Stat label="Başarısız kavrama" value="−0.2" color={C.fail} sub="boş, engelli, geniş, üstü kapalı ya da kaydı" />
      </div>

      <Section num="4" title="Bölüm ve getiri">
        Bir bölüm, bir kutuyu boşaltma denemesidir. Başarılı her alım yükseklik haritasını değiştirir ve alttaki parçaları
        açığa çıkarır. Bu yüzden kararlar birbirine bağlıdır ve ajan indirimli toplam ödülü (getiri) en büyütmeye çalışır:
      </Section>
      <div className="grid g2">
        <TeX>{"G_t = r_{t+1} + \\gamma\\, r_{t+2} + \\gamma^2 r_{t+3} + \\dots = \\sum_{k=0}^{\\infty} \\gamma^k\\, r_{t+k+1}, \\qquad \\gamma = 0.5"}</TeX>
        <Card title="Bölüm ne zaman biter?">
          Kutu tamamen boşalınca (<b>terminal</b> durum) ya da <b>⌈2.5 × parça sayısı⌉</b> deneme yapılınca (kesme). Kesmede
          gelecek değer hâlâ hesaba katılır, terminal durumda katılmaz.
        </Card>
      </div>
    </div>
  );
}
