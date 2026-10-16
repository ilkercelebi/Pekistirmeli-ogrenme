import { useEffect, useRef, useState } from "react";
import { api, type GraspEvent, type Part, type Scenario } from "../api";
import { Banner, Card, Chip, Section, Stat } from "../components/ui";
import { useMeta } from "../hooks";
import RobotCell from "../three/Scene";
import { C, OUTCOME } from "../theme";

function ScenarioPlayer() {
  const meta = useMeta("parallel_jaw");
  const [list, setList] = useState<Scenario[]>([]);
  const [idx, setIdx] = useState(0);
  const [parts, setParts] = useState<Part[]>([]);
  const [event, setEvent] = useState<GraspEvent | null>(null);
  const [key, setKey] = useState(0);
  const [last, setLast] = useState<GraspEvent | null>(null);
  const [auto, setAuto] = useState(true);
  const timer = useRef<number>();

  useEffect(() => {
    api.scenarios().then(setList).catch(() => setList([]));
  }, []);

  const sc = list[idx];
  useEffect(() => {
    if (!sc) return;
    setParts(sc.parts);
    setLast(null);
    setEvent(null);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      setEvent(sc.event);
      setKey((k) => k + 1);
    }, 700);
    return () => window.clearTimeout(timer.current);
  }, [sc]);

  const done = () => {
    if (!sc) return;
    if (sc.event.outcome === "success") setParts(sc.parts.map((p) => (p.id === sc.event.part ? { ...p, alive: false } : p)));
    setLast(sc.event);
    setEvent(null);
    if (auto) timer.current = window.setTimeout(() => setIdx((i) => (i + 1) % list.length), 2600);
  };

  if (!meta || !sc) return <div className="viewport" style={{ height: 560 }} />;

  return (
    <div className="split">
      <RobotCell meta={meta} parts={parts} depth={sc.depth} event={event} eventKey={key} onEventDone={done} speed={0.9}
        height={560} signal={last ? last.outcome : "idle"} view="close">
        <div className="overlay-tl hud"><span className="dot" style={{ background: C.accent }} />Senaryo {idx + 1} / {list.length}</div>
        <div className="overlay-bl hud">Sürükle: döndür · Tekerlek: yakınlaştır</div>
      </RobotCell>
      <div className="stack">
        <div className="seg" style={{ alignSelf: "flex-start" }}>
          {list.map((s, i) => (
            <button key={s.key} className={i === idx ? "on" : ""} onClick={() => { setAuto(false); setIdx(i); }}>
              {i + 1}
            </button>
          ))}
          <button className={auto ? "on" : ""} onClick={() => setAuto((a) => !a)}>{auto ? "Otomatik ●" : "Otomatik ○"}</button>
        </div>
        <Card kicker={`Senaryo ${idx + 1}`} title={sc.title} tone={OUTCOME[sc.event.outcome].color}>
          {sc.explanation}
        </Card>
        <Banner event={last} />
        <button className="btn" onClick={() => { setAuto(false); setParts(sc.parts); setLast(null); setEvent(sc.event); setKey((k) => k + 1); }}
          disabled={!!event}>
          Tekrar oynat
        </button>
        <p className="note">
          Bu dört durum elle kuruldu ve kayma olasılığı kapatıldı; amaç sonuç kurallarını göstermek. Simülasyon sayfasında
          kutular rastgele üretilir ve kararları ajan verir.
        </p>
      </div>
    </div>
  );
}

export default function Intro() {
  return (
    <div>
      <section className="hero fade-in">
        <div className="kicker">Ödev 8 · Pekiştirmeli öğrenme</div>
        <h1>Düzensiz kutudan parça alma:<br />nereden ve hangi açıyla tutmalı?</h1>
        <p className="lead">
          Robot, kutuya rastgele yığılmış parçaları tek tek almak zorunda. Yanlış seçilen bir nokta tutucuyu kutu duvarına
          çarptırır ya da parçayı düşürür. Bu uygulama, doğru kararı <b style={{ color: C.text }}>deneme-yanılma ile öğrenen</b>,
          sinir ağı kullanmayan, tablo tabanlı bir pekiştirmeli öğrenme ajanını adım adım anlatıyor.
        </p>
        <div className="row" style={{ marginTop: 20 }}>
          <Chip color={C.accent}>Q-learning · SARSA</Chip>
          <Chip color={C.info}>Derinlik kamerası → yükseklik haritası</Chip>
          <Chip color={C.success}>Derin öğrenme yok</Chip>
        </div>
      </section>

      <div className="grid g4">
        <Stat label="Kutudaki parça" value="6–14" sub="rastgele yığın" />
        <Stat label="Aday eylem / adım" value="6.400" sub="40 × 40 nokta × 4 açı" />
        <Stat label="Q tablosu" value="1.296" sub="hücre (parmaklı tutucu)" color={C.accent} />
        <Stat label="Sinir ağı" value="0" sub="parametre öğrenilmiyor" color={C.success} />
      </div>

      <Section num="1" title="Bir kavrama nasıl sonuçlanır?">
        Aşağıdaki sahne gerçek simülasyonun 3D görüntüsü. Kartezyen robot, kutunun üstündeki derinlik kamerasının
        gördüğüne göre bir nokta ve bilek açısı seçer, iner ve parçayı kavramaya çalışır.
      </Section>
      <ScenarioPlayer />

      <Section num="2" title="Endüstriyel problem">
        Kutudan parça alma (bin picking) üretim ve lojistikte en zor otomasyon işlerinden biridir. Bir kavrama üç şekilde ters gidebilir:
      </Section>
      <div className="grid g3">
        <Card kicker="−1 ödül" title="Kutuya çarpma" tone={C.collision}>
          Parça duvarın dibindeyse bazı açılarda parmaklardan biri <b>duvarın üstüne iner</b>. Tutucu ve kutu zarar görür, hat durur.
        </Card>
        <Card kicker="−0.2 ödül" title="Başarısız kavrama" tone={C.fail}>
          Boş bir noktaya inmek, parmakların komşu parçaya takılması, parçanın tutucu açıklığından <b>geniş</b> olması ya da
          üstünde <b>başka parça</b> bulunması.
        </Card>
        <Card kicker="−0.2 ödül" title="Kayıp düşme" tone={C.fail}>
          Parça tutulur ama yanlış açı ya da ağırlık merkezinden uzak tutma yüzünden <b>kalkarken kayar</b>. Olasılığı kavrama
          kalitesine bağlıdır.
        </Card>
      </div>

      <Section num="3" title="Bu uygulamada neler var?" />
      <div className="grid g3">
        {[
          ["mdp", "MDP tasarımı", "Durum, eylem ve ödül nasıl tanımlandı? Kamera ne görüyor, ajan ne seçiyor?"],
          ["yontem", "Yöntem", "6.400 aday eylem birkaç ayrık özelliğe nasıl indirgeniyor ve Q tablosu nasıl öğreniliyor?"],
          ["simulasyon", "Simülasyon", "Ajanı canlı izle ya da kavrama noktasını kendin seç."],
          ["karsilastirma", "Karşılaştırma", "Rastgele, sezgisel ve öğrenen ajanlar aynı test kutularında."],
          ["egitim", "Eğitim ve sonuçlar", "Ajanı tarayıcıdan eğit, öğrenme eğrilerini ve öğrendiği kuralları incele."],
        ].map(([id, t, d], i) => (
          <a key={id} href={`#/${id}`} style={{ textDecoration: "none" }}>
            <Card kicker={`Adım ${i + 2}`} title={t}>{d}</Card>
          </a>
        ))}
      </div>
    </div>
  );
}
