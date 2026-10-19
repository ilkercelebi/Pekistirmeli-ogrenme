// Kavrama animasyonunun saf (React/Three.js'siz) planlama mantığı.
// Robot.tsx bunu her karede çağırır; plan.test.ts aynı fonksiyonları doğrudan test eder.
//
// Fiziksel model (paralel tutucu):
//   1. Parmaklar açık iner. Parmaklar kapanırken parça, ilk parmak ona değene kadar yerinde durur.
//   2. Temastan sonra parça iki parmağın ortasına itilir ve yüzleri parmaklara paralel olacak şekilde döner.
//   3. Kaldırılan parça tutucuya göre sabittir: tutucu dönünce parça da onunla döner.
//   4. Tepside robot bileğini, parça tepsideki yönüne gelecek şekilde çevirir, alçalır ve bırakır;
//      parça tam olarak tepsideki yerine oturur (bırakıldıktan sonra yer/yön atlaması yoktur).
//
// Koordinatlar: dünya (x, z) yatay düzlem, y yukarı. Tutucunun kapanma yönü u = (cos θ, sin θ);
// Three.js'te tutucu dönüşü rotation.y = −θ. Parça yönü `yaw` = rotation.y (açısı 0 çizilen parça için).
import type { GraspEvent, Meta, Part } from "../api";
import { HOME, ease, lerp, lerpAngle, toWorld } from "./util";

export type Pose = { x: number; y: number; z: number; th: number; open: number };
export type Key = { t: number; pose: Pose; ease?: (t: number) => number };
export type Vec3 = [number, number, number];
export interface PartPose { pos: Vec3; yaw: number }

export interface Plan {
  keys: Key[];
  total: number;
  part?: Part;
  /** success: tepsiye taşınır · slip: kalkarken düşer · tug: üstü kapalı, kalkmaz · none: parçaya dokunulmaz */
  mode: "success" | "slip" | "tug" | "none";
  gripStart: number;          // parmakların kapanmaya başladığı an
  attach: number;             // parmakların kapandığı an (bu andan sonra parça tutucuya bağlı)
  release: number;            // success: tepsiye bırakma anı
  slip: number;               // slip: parçanın düşmeye başladığı an
  shake: [number, number] | null;
  outcomeAt: number;          // LED'in sonucu göstermeye başladığı an
  trayTarget?: [number, number, number, number];
  home: PartPose;             // parçanın kutudaki pozu
  squeezed: PartPose;         // parmaklar kapandıktan sonraki pozu (ortalanmış, hizalanmış)
  contactInner: number;       // ilk temas anındaki parmak iç yüzü uzaklığı
  finalInner: number;         // parmaklar kapandığında iç yüz uzaklığı
  fingerT: number;            // parmak kalınlığı
  local: { pos: Vec3; yaw: number };  // bağlıyken tutucu çerçevesindeki poz
  landing?: PartPose;         // success: tepsideki son poz
}

const TAU_EPS = 1e-6;

/** Three.js Y dönüşü: yerel (x, z) → dünya. */
export function rotY(x: number, z: number, a: number): [number, number] {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [x * c + z * s, -x * s + z * c];
}

export function poseAt(plan: Plan, t: number): Pose {
  const k = plan.keys;
  if (t <= k[0].t) return k[0].pose;
  for (let i = 1; i < k.length; i++) {
    if (t <= k[i].t) {
      const a = k[i - 1];
      const b = k[i];
      const f = (b.ease ?? ease.inOut)((t - a.t) / (b.t - a.t || 1));
      return {
        x: lerp(a.pose.x, b.pose.x, f), y: lerp(a.pose.y, b.pose.y, f), z: lerp(a.pose.z, b.pose.z, f),
        th: lerpAngle(a.pose.th, b.pose.th, f), open: lerp(a.pose.open, b.pose.open, f),
      };
    }
  }
  return k[k.length - 1].pose;
}

/** Tutucu merkezinden u yönünde geçen doğrunun parçayla kesişimi: [t1, t2] (u boyunca uzaklık), yoksa null. */
export function chord(part: Part, worldCenter: (p: Part) => [number, number], cx: number, cz: number, ux: number, uz: number) {
  const [px, pz] = worldCenter(part);
  const dx = cx - px;
  const dz = cz - pz;
  if (part.kind === "disc") {
    const r = part.length / 2;
    const b = dx * ux + dz * uz;
    const perp2 = dx * dx + dz * dz - b * b;
    if (perp2 > r * r) return null;
    const s = Math.sqrt(r * r - perp2);
    return [-b - s, -b + s] as [number, number];
  }
  const ax = Math.cos(part.angle);
  const az = Math.sin(part.angle);
  const d = [dx * ax + dz * az, -dx * az + dz * ax];
  const u = [ux * ax + uz * az, -ux * az + uz * ax];
  const half = [part.length / 2, part.width / 2];
  let lo = -Infinity;
  let hi = Infinity;
  for (let i = 0; i < 2; i++) {
    if (Math.abs(u[i]) < TAU_EPS) {
      if (Math.abs(d[i]) > half[i]) return null;
      continue;
    }
    const a = (-half[i] - d[i]) / u[i];
    const b = (half[i] - d[i]) / u[i];
    lo = Math.max(lo, Math.min(a, b));
    hi = Math.min(hi, Math.max(a, b));
  }
  return lo <= hi ? ([lo, hi] as [number, number]) : null;
}

export function buildPlan(meta: Meta, ev: GraspEvent, parts: Part[], start: Pose, trayTarget?: [number, number, number, number]): Plan {
  const g = meta.gripper;
  const vacuum = g.kind === "vacuum";
  const H = (meta.grid.size - 1) / 2;
  const center = (p: Part): [number, number] => [p.cx - H, p.cy - H];
  const [tx, tz] = toWorld(meta, ev.action.row, ev.action.col);
  const th = ((ev.action.theta_deg ?? meta.angles_deg[ev.action.angle_idx]) * Math.PI) / 180;
  const ux = Math.cos(th);
  const uz = Math.sin(th);
  const ft = g.finger_thickness;
  const open0 = g.opening / 2 + ft / 2;
  const hoverY = meta.grid.wall_height + 12;
  const stopY = ev.outcome === "collision" ? meta.grid.wall_height : ev.anim.stop_z;
  const part = ev.part != null ? parts.find((p) => p.id === ev.part) : undefined;

  // ------------------------------------------------ parçanın kutudaki pozu ve sıkıştırılmış pozu
  const home: PartPose = part
    ? { pos: [center(part)[0], part.base + part.thickness / 2, center(part)[1]], yaw: -part.angle }
    : { pos: [tx, 0, tz], yaw: 0 };
  let squeezed: PartPose = { pos: [...home.pos] as Vec3, yaw: home.yaw };
  let contactInner = g.opening / 2;
  let finalInner = g.opening / 2;
  let mode: Plan["mode"] = "none";
  if (part && ev.code === "ok") mode = "success";
  else if (part && ev.code === "slip") mode = "slip";
  else if (part && ev.code === "covered") mode = "tug";

  let close = open0;
  if (!vacuum) {
    const c = part ? chord(part, center, tx, tz, ux, uz) : null;
    if (part && c && mode !== "none") {
      contactInner = Math.min(g.opening / 2, Math.max(c[1], -c[0]));
      if (mode === "tug") {
        // üstü kapalı parça kıpırdamaz: parmaklar yalnızca değer
        finalInner = contactInner;
      } else {
        // dik doğrultudaki kaçıklık korunur, kapanma doğrultusunda parça ortalanır
        const nx = -uz;
        const nz = ux;
        const e = (home.pos[0] - tx) * nx + (home.pos[2] - tz) * nz;
        let delta = 0;
        let dim = c[1] - c[0];
        if (part.kind === "box" && mode === "success") {
          // en yakın yüz parmaklara paralel olacak şekilde döner (kayan parça hizalanamadığı için kayar)
          const alpha = th - part.angle;
          const k = Math.round(alpha / (Math.PI / 2));
          delta = alpha - k * (Math.PI / 2);
          dim = k % 2 === 0 ? part.length : part.width;
          if (dim > g.opening - 0.2) {
            delta -= Math.sign(delta || 1) * (Math.PI / 2);
            dim = k % 2 === 0 ? part.width : part.length;
          }
        } else if (part.kind === "disc") {
          dim = 2 * Math.sqrt(Math.max(0, (part.length / 2) ** 2 - e * e));
        }
        finalInner = Math.min(contactInner, dim / 2);
        squeezed = { pos: [tx + e * nx, home.pos[1], tz + e * nz], yaw: home.yaw - delta };
      }
      close = finalInner + ft / 2;
    } else if (ev.code === "empty") close = ft / 2 + 0.4;
    else close = open0 * 0.93;
  }

  // ------------------------------------------------ robot yolu
  const keys: Key[] = [];
  let t = 0;
  const push = (d: number, p: Partial<Pose>, e?: (t: number) => number) => {
    const prev = keys.length ? keys[keys.length - 1].pose : start;
    t += d;
    keys.push({ t, pose: { ...prev, ...p }, ease: e });
  };
  keys.push({ t: 0, pose: start });
  push(1.05, { x: tx, z: tz, y: hoverY, th, open: open0 });
  push(0.75, { y: stopY }, ev.outcome === "collision" ? ease.in : ease.inOut);

  const plan: Plan = {
    keys, total: 0, part, mode, gripStart: t, attach: t, release: 0, slip: 0, shake: null, outcomeAt: t, trayTarget,
    home, squeezed, contactInner, finalInner, fingerT: ft, local: { pos: [0, 0, 0], yaw: 0 },
  };

  if (ev.outcome === "collision") {
    plan.mode = "none";
    plan.shake = [t, t + 0.55];
    push(0.55, {});
    push(0.6, { y: hoverY });
    push(0.95, { ...HOME, th: 0, open: open0 });
    plan.total = t;
    return plan;
  }

  push(vacuum ? 0.3 : 0.45, { open: close }, ease.out);   // kavrama: parmaklar kapanır / vakum çeker
  plan.attach = t;
  plan.outcomeAt = t;

  // bağlanma anında parçanın tutucu çerçevesindeki pozu
  const at = poseAt(plan, plan.attach);
  const [lx, lz] = rotY(squeezed.pos[0] - at.x, squeezed.pos[2] - at.z, th);   // −τ = +θ
  plan.local = { pos: [lx, squeezed.pos[1] - at.y, lz], yaw: squeezed.yaw + th };

  if (mode === "success") {
    push(0.85, { y: hoverY });
    const slot = trayTarget ?? [HOME.x, 1, HOME.z, 0];
    // tepsideki yön: yaw = −θ + local.yaw ≡ slot yönü (mod π, parçalar 180° simetrik)
    const thEnd = lerpAngle(at.th, plan.local.yaw - slot[3], 1);
    const [wx, wz] = rotY(plan.local.pos[0], plan.local.pos[2], -thEnd);
    const drop = 0.35;
    const placeY = slot[1] + drop - plan.local.pos[1];
    push(1.05, { x: slot[0] - wx, z: slot[2] - wz, y: Math.max(placeY + 6, hoverY - 6), th: thEnd });
    push(0.55, { y: placeY }, ease.out);
    plan.release = t;
    push(0.28, { open: open0 });
    push(0.45, { y: placeY + 8 });
    push(0.9, { x: HOME.x, y: HOME.y, z: HOME.z, th: 0 });
    plan.landing = { pos: [slot[0], slot[1], slot[2]], yaw: -thEnd + plan.local.yaw };
  } else {
    if (mode === "slip") plan.slip = t + 0.32;
    // başarısız kavramada (geniş parça, engel, üstü kapalı, boş nokta) tutucu bir an bekler: ne olduğu görülsün
    if (ev.outcome === "fail" && mode !== "slip") push(0.55, {});
    push(0.85, { y: hoverY });
    push(0.3, { open: open0 });
    push(0.9, { ...HOME, th: 0 });
  }
  plan.total = t;
  return plan;
}

const smooth = (f: number) => f * f * (3 - 2 * f);

function lerpPose(a: PartPose, b: PartPose, f: number): PartPose {
  return { pos: [lerp(a.pos[0], b.pos[0], f), lerp(a.pos[1], b.pos[1], f), lerp(a.pos[2], b.pos[2], f)], yaw: lerp(a.yaw, b.yaw, f) };
}

/** Parmaklar kapanırken parçanın pozu: temasa kadar yerinde, sonra ortalanır ve hizalanır. */
function squeezePose(plan: Plan, t: number): PartPose {
  const inner = poseAt(plan, t).open - plan.fingerT / 2;
  const span = plan.contactInner - plan.finalInner;
  const f = span > 1e-3 ? Math.min(1, Math.max(0, (plan.contactInner - inner) / span)) : t >= plan.attach ? 1 : 0;
  return lerpPose(plan.home, plan.squeezed, smooth(f));
}

function heldPose(plan: Plan, t: number): PartPose {
  const q = poseAt(plan, t);
  const [wx, wz] = rotY(plan.local.pos[0], plan.local.pos[2], -q.th);
  return { pos: [q.x + wx, q.y + plan.local.pos[1], q.z + wz], yaw: -q.th + plan.local.yaw };
}

function fallPose(from: PartPose, to: PartPose, f: number, bounceH = 0.6): PartPose {
  const e = ease.in(Math.min(1, f));
  const p = lerpPose(from, to, e);
  if (f > 0.78 && f < 1) p.pos[1] += Math.sin(((f - 0.78) / 0.22) * Math.PI) * bounceH;
  return p;
}

/** Parçanın dünya pozu; null = olay parçaya dokunmuyor (parça sahnedeki yerinde çizilir). */
export function carriedPose(plan: Plan, t: number): PartPose | null {
  if (!plan.part || plan.mode === "none") return null;
  if (t < plan.attach) return plan.mode === "tug" ? plan.home : squeezePose(plan, t);
  if (plan.mode === "tug") {
    // tutucu çeker, parça hafifçe esner ama kalkamaz
    const f = (t - plan.attach) / 0.4;
    const pose: PartPose = { pos: [...plan.home.pos] as Vec3, yaw: plan.home.yaw };
    if (f > 0 && f < 1) pose.pos[1] += Math.sin(f * Math.PI) * 0.45;
    return pose;
  }
  if (plan.mode === "success") {
    if (t < plan.release) return heldPose(plan, t);
    const landing = plan.landing ?? heldPose(plan, plan.release);
    return fallPose(heldPose(plan, plan.release), landing, (t - plan.release) / 0.16, 0.15);
  }
  // kayma: parça kalkarken tutucudan kayar ve kutudaki yerine düşer
  if (t < plan.slip) return heldPose(plan, t);
  return fallPose(heldPose(plan, plan.slip), plan.home, (t - plan.slip) / 0.42);
}
