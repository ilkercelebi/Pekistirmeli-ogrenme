import { lazy, Suspense } from "react";
import { useHash } from "./hooks";

const Intro = lazy(() => import("./pages/Intro"));
const Mdp = lazy(() => import("./pages/Mdp"));
const Method = lazy(() => import("./pages/Method"));
const Demo = lazy(() => import("./pages/Demo"));
const Compare = lazy(() => import("./pages/Compare"));
const Training = lazy(() => import("./pages/Training"));

export const PAGES = [
  { id: "giris", title: "Problem", Comp: Intro },
  { id: "mdp", title: "MDP tasarımı", Comp: Mdp },
  { id: "yontem", title: "Yöntem", Comp: Method },
  { id: "simulasyon", title: "Simülasyon", Comp: Demo },
  { id: "karsilastirma", title: "Karşılaştırma", Comp: Compare },
  { id: "egitim", title: "Eğitim ve sonuçlar", Comp: Training },
];

export default function App() {
  const [hash] = useHash();
  const page = PAGES.find((p) => p.id === hash) ?? PAGES[0];
  const Comp = page.Comp;
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <div className="brand-mark">Q</div>
          <div>
            <b>Bin Picking RL</b>
            <span>Ödev 8 · Optimal kavrama seçimi</span>
          </div>
        </div>
        <nav className="nav">
          {PAGES.map((p, i) => (
            <a key={p.id} href={`#/${p.id}`} className={p.id === page.id ? "on" : ""}>
              <span className="n">{i + 1}</span>
              {p.title}
            </a>
          ))}
        </nav>
        <div className="side-foot">
          <b>Tablo tabanlı pekiştirmeli öğrenme.</b> Q-learning, SARSA ve Double Q-learning; sinir ağı yok.
          Simülasyon Python'da çalışır, arayüz React + Three.js.
        </div>
      </aside>
      <main className="main">
        <Suspense fallback={<div className="note">Yükleniyor…</div>}>
          <Comp key={page.id} />
        </Suspense>
      </main>
    </div>
  );
}
