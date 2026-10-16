// Kavrama animasyonunun saf (React/Three.js'siz) planlama mantığı.
// Robot.tsx bunu her karede çağırır; plan.test.ts aynı fonksiyonları doğrudan test eder.
import type { GraspEvent, Meta, Part } from "../api";
import { HOME, ease, extentAlong, lerp, lerpAngle, toWorld } from "./util";

export type Pose = { x: number; y: number; z: number; th: number; open: number };
export type Key = { t: number; pose: Pose; ease?: (t: number) => number };

export interface Plan {
  keys: Key[];
  total: number;
  part?: Part;
  mode: "success" | "slip" | "none";
  attach: number;
  offset: [number, number, number];
  release: number;           // success: tepsiye bırakma anı
  slip: number;              // slip: parçanın düşmeye başladığı an
  shake: [number, number] | null;
  outcomeAt: number;         // LED'in sonucu göstermeye başladığı an
  trayTarget?: [number, number, number, number];
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

export function buildPlan(meta: Meta, ev: GraspEvent, parts: Part[], start: Pose, trayTarget?: [number, number, number, number]): Plan {
  const g = meta.gripper;
  const [tx, tz] = toWorld(meta, ev.action.row, ev.action.col);
  const th = ((ev.action.theta_deg ?? meta.angles_deg[ev.action.angle_idx]) * Math.PI) / 180;
  const open0 = g.opening / 2 + g.finger_thickness / 2;
  const hoverY = meta.grid.wall_height + 12;
  const stopY = ev.outcome === "collision" ? meta.grid.wall_height : ev.anim.stop_z;
  const part = ev.part != null ? parts.find((p) => p.id === ev.part) : undefined;

  let close = open0;
  if (g.kind !== "vacuum") {
    if (part && ["ok", "slip", "covered"].includes(ev.code)) {
      close = Math.min(open0, extentAlong(part, th) / 2 + g.finger_thickness / 2);
    } else if (ev.code === "empty") close = g.finger_thickness / 2 + 0.4;
    else close = open0 * 0.93;
  }

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
    keys, total: 0, part, mode: "none", attach: 0, offset: [0, 0, 0], release: 0, slip: 0, shake: null,
    outcomeAt: t, trayTarget,
  };

  if (ev.outcome === "collision") {
    plan.shake = [t, t + 0.55];
    push(0.55, {});
    push(0.6, { y: hoverY });
    push(0.95, { ...HOME, th: 0, open: open0 });
  } else {
    push(0.35, { open: close });
    plan.outcomeAt = t;
    plan.attach = t;
    if (ev.code === "ok" && part) {
      plan.mode = "success";
      push(0.85, { y: hoverY });
      push(1.0, { x: trayTarget ? trayTarget[0] : HOME.x, z: trayTarget ? trayTarget[2] : HOME.z, y: hoverY - 4 });
      plan.release = t + 0.12;
      push(0.32, { open: open0 });
      push(0.9, { x: HOME.x, y: HOME.y, z: HOME.z, th: 0 });
    } else if (ev.code === "slip" && part) {
      plan.mode = "slip";
      plan.slip = t + 0.42;
      push(0.85, { y: hoverY });
      push(0.3, { open: open0 });
      push(0.9, { ...HOME, th: 0 });
    } else {
      push(0.85, { y: hoverY });
      push(0.3, { open: open0 });
      push(0.9, { ...HOME, th: 0 });
    }
    if (part) {
      const [px, pz] = [part.cx - (meta.grid.size - 1) / 2, part.cy - (meta.grid.size - 1) / 2];
      const at = poseAt({ ...plan, keys }, plan.attach);
      plan.offset = [px - at.x, part.base + part.thickness / 2 - at.y, pz - at.z];
    }
  }
  plan.total = t;
  return plan;
}

/** Taşınan parçanın dünya konumu; null = parça henüz kutudaki yerinde. */
export function carriedPose(plan: Plan, t: number): [number, number, number] | null {
  if (!plan.part || plan.mode === "none" || t < plan.attach) return null;
  const held = (tt: number): [number, number, number] => {
    const q = poseAt(plan, tt);
    return [q.x + plan.offset[0], q.y + plan.offset[1], q.z + plan.offset[2]];
  };
  const fall = (from: [number, number, number], to: [number, number, number], f: number) => {
    const e = ease.in(Math.min(1, f));
    const bounce = f > 0.78 && f < 1 ? Math.sin(((f - 0.78) / 0.22) * Math.PI) * 0.7 : 0;
    return [lerp(from[0], to[0], e), lerp(from[1], to[1], e) + bounce, lerp(from[2], to[2], e)] as [number, number, number];
  };
  if (plan.mode === "success") {
    if (t < plan.release) return held(t);
    const to = plan.trayTarget ? (plan.trayTarget.slice(0, 3) as [number, number, number]) : held(plan.release);
    return fall(held(plan.release), to, (t - plan.release) / 0.42);
  }
  // kayıp düşme: parça kalkarken tutucudan kayar ve eski yerine düşer
  if (t < plan.slip) return held(t);
  return fall(held(plan.slip), held(plan.attach), (t - plan.slip) / 0.45);
}
