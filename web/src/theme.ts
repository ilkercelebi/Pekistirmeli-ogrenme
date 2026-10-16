import type { Outcome, Tone } from "./api";

export const C = {
  bg: "#0a0d11",
  panel: "#11161d",
  panel2: "#171e27",
  border: "#232d3a",
  text: "#e8edf2",
  muted: "#8a96a3",
  accent: "#ff7a1a",
  success: "#3fb950",
  collision: "#f85149",
  fail: "#e3b341",
  info: "#58a6ff",
  wall: "#2f5f93",
};

export const OUTCOME: Record<Outcome, { label: string; reward: string; color: string }> = {
  success: { label: "BAŞARILI", reward: "+1", color: C.success },
  collision: { label: "ÇARPIŞMA", reward: "−1", color: C.collision },
  fail: { label: "BAŞARISIZ", reward: "−0.2", color: C.fail },
};

export const TONE: Record<Tone, string> = { good: C.success, mid: C.fail, bad: C.collision };

export const FEATURE_TITLE: Record<string, string> = {
  yükseklik: "Yükseklik",
  düzlük: "Düzlük",
  yakında_yüksek: "Yakında daha yüksek",
  duvar: "Duvar",
  parmak_boşluğu: "Parmak boşluğu",
  genişlik: "Genişlik",
  ortalı: "Ortalı",
  sızdırmazlık: "Sızdırmazlık",
};

// Yükseklik renk skalası (zemin koyu petrol → yüksek parça açık kehribar)
const HEIGHT_STOPS: [number, [number, number, number]][] = [
  [0.0, [15, 27, 38]],
  [0.18, [23, 58, 74]],
  [0.45, [31, 122, 128]],
  [0.72, [224, 163, 58]],
  [1.0, [255, 241, 193]],
];

const Q_STOPS: [number, [number, number, number]][] = [
  [0.0, [248, 81, 73]],
  [0.4, [70, 45, 30]],
  [0.5, [27, 35, 45]],
  [0.6, [30, 66, 40]],
  [1.0, [63, 185, 80]],
];

function ramp(stops: [number, [number, number, number]][], t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    const [t0, c0] = stops[i - 1];
    if (v <= t1) {
      const f = (v - t0) / (t1 - t0 || 1);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

export const heightColor = (h: number, max: number) => ramp(HEIGHT_STOPS, h / max);
export const qColor = (q: number) => ramp(Q_STOPS, (q + 1) / 2);
export const rgb = (c: [number, number, number]) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

export const fmt = (v: number | null | undefined, d = 2) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));
export const pct = (v: number | null | undefined) => (v == null ? "—" : `%${Math.round(v * 100)}`);
