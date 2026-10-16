import type { Meta, Part } from "../api";

// Izgara (satır, sütun) → dünya (x, z). 1 birim = 1 cm.
export const toWorld = (meta: Meta, row: number, col: number): [number, number] => {
  const h = (meta.grid.size - 1) / 2;
  return [col - h, row - h];
};

export const TRAY = { x: 33, z: 0, w: 14, d: 26, h: 5 };
export const HOME = { x: 0, y: 44, z: -31 };

// Gerçek endüstriyel parça yüzeyleri: fırçalanmış çelik, eloksal alüminyum, pirinç, siyah POM
const FINISHES = [
  { color: "#a9b2bb", metalness: 0.9, roughness: 0.32 },
  { color: "#5d6873", metalness: 0.85, roughness: 0.4 },
  { color: "#b8955a", metalness: 0.9, roughness: 0.3 },
  { color: "#3f72b0", metalness: 0.7, roughness: 0.35 },
  { color: "#c4692d", metalness: 0.7, roughness: 0.38 },
  { color: "#22262b", metalness: 0.15, roughness: 0.55 },
  { color: "#d2d7dc", metalness: 0.95, roughness: 0.22 },
];

export const finishFor = (p: Part) => FINISHES[(p.id * 7 + (p.kind === "disc" ? 3 : 0)) % FINISHES.length];

export const ease = {
  inOut: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out: (t: number) => 1 - Math.pow(1 - t, 3),
  in: (t: number) => t * t * t,
};

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function lerpAngle(a: number, b: number, t: number) {
  // kavrama açıları 180° periyotludur
  let d = (((b - a) % Math.PI) + Math.PI) % Math.PI;
  if (d > Math.PI / 2) d -= Math.PI;
  return a + d * t;
}

/** Parçanın θ kapanma yönündeki genişliği (parmakların kapanacağı mesafe). */
export function extentAlong(p: Part, theta: number) {
  if (p.kind === "disc") return p.length;
  const d = theta - p.angle;
  return Math.abs(p.length * Math.cos(d)) + Math.abs(p.width * Math.sin(d));
}
