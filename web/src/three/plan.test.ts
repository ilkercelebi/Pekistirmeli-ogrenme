// Kavrama animasyonu planı: her sonuç türü için robotun ve taşınan parçanın hareketi.
import { describe, expect, it } from "vitest";
import type { GraspEvent, Meta, Part } from "../api";
import { buildPlan, carriedPose, poseAt, type Plan } from "./plan";
import { HOME, extentAlong, lerpAngle } from "./util";

const meta = (kind: "parallel_jaw" | "vacuum" = "parallel_jaw"): Meta => ({
  gripper: { kind, opening: 9, finger_thickness: 1, finger_width: 3, grasp_depth: 1.5, cup_radius: 2 },
  grid: { size: 44, wall_thickness: 2, wall_height: 20, max_stack_height: 15 },
  angles_deg: kind === "vacuum" ? [0] : [0, 45, 90, 135],
  rewards: { success: 1, collision: -1, fail: -0.2 },
  features: [], n_states: 1296, dims: [], n_parts: [6, 14], candidates: 6400,
});

const part: Part = { id: 3, kind: "box", cx: 21, cy: 21, length: 10, width: 4, angle: 0, base: 0, thickness: 3, alive: true };
const start = { ...HOME, th: 0, open: 5 };
const tray: [number, number, number, number] = [33, 1.6, -8.2, 0];
const H = (44 - 1) / 2;
const original: [number, number, number] = [part.cx - H, part.base + part.thickness / 2, part.cy - H];

function event(outcome: GraspEvent["outcome"], code: string, extra: Partial<GraspEvent> = {}): GraspEvent {
  return {
    action: { angle_idx: 2, row: 21, col: 21, theta_deg: 90 },
    reward: outcome === "success" ? 1 : outcome === "collision" ? -1 : -0.2,
    outcome, code, reason: code, part: part.id, anim: { top_z: 3, stop_z: 1.5 }, ...extra,
  };
}

const sample = (plan: Plan, n = 400) => Array.from({ length: n + 1 }, (_, i) => (plan.total * i) / n);
const near = (a: number[], b: number[], eps = 1e-6) => a.every((v, i) => Math.abs(v - b[i]) < eps);

describe("kavrama planı", () => {
  it("başarılı kavramada parça tepsiye bırakılır, robot eve döner", () => {
    const plan = buildPlan(meta(), event("success", "ok"), [part], start, tray);
    expect(plan.mode).toBe("success");
    // kavrama anında parça kutudaki yerinden alınır (sıçrama yok)
    expect(near(carriedPose(plan, plan.attach)!, original)).toBe(true);
    // kaldırılırken yükselir
    expect(carriedPose(plan, plan.attach + 0.8)![1]).toBeGreaterThan(original[1] + 5);
    // sonunda tepsideki yerindedir
    expect(near(carriedPose(plan, plan.total)!, tray.slice(0, 3))).toBe(true);
    const end = poseAt(plan, plan.total);
    expect([end.x, end.y, end.z]).toEqual([HOME.x, HOME.y, HOME.z]);
  });

  it("parmaklar parçanın kısa kenarına kadar kapanır", () => {
    const plan = buildPlan(meta(), event("success", "ok"), [part], start, tray);
    const closed = poseAt(plan, plan.attach).open;
    // θ = 90°: parça x ekseninde uzun, kapanma yönündeki genişlik 4 cm → parmak merkezi 2 + 0.5
    expect(closed).toBeCloseTo(extentAlong(part, Math.PI / 2) / 2 + 0.5, 6);
    expect(closed).toBeCloseTo(2.5, 6);
  });

  it("kayan parça kalkar, sonra eski yerine düşer", () => {
    const plan = buildPlan(meta(), event("fail", "slip"), [part], start);
    expect(plan.mode).toBe("slip");
    const peak = Math.max(...sample(plan).map((t) => carriedPose(plan, t)?.[1] ?? -Infinity));
    expect(peak).toBeGreaterThan(original[1] + 3);
    expect(near(carriedPose(plan, plan.total)!, original)).toBe(true);
  });

  it("çarpışmada tutucu duvar üstünde durur, sarsılır ve parça taşınmaz", () => {
    const plan = buildPlan(meta(), event("collision", "wall", { part: null, anim: { top_z: 0, stop_z: 20 } }), [part], start);
    expect(plan.shake).not.toBeNull();
    const [s0, s1] = plan.shake!;
    expect(s1 - s0).toBeGreaterThan(0.3);
    const minY = Math.min(...sample(plan).map((t) => poseAt(plan, t).y));
    expect(minY).toBeCloseTo(20, 6); // duvarın altına hiç inmez
    expect(sample(plan).every((t) => carriedPose(plan, t) === null)).toBe(true);
    const end = poseAt(plan, plan.total);
    expect([end.x, end.y, end.z]).toEqual([HOME.x, HOME.y, HOME.z]);
  });

  it("boş noktada parmaklar tamamen kapanır, üstü kapalı parçada parça yerinden oynamaz", () => {
    const empty = buildPlan(meta(), event("fail", "empty", { part: null }), [part], start);
    expect(poseAt(empty, empty.attach).open).toBeCloseTo(0.9, 6);
    expect(empty.mode).toBe("none");
    const covered = buildPlan(meta(), event("fail", "covered"), [part], start);
    expect(covered.mode).toBe("none");
    expect(sample(covered).every((t) => carriedPose(covered, t) === null)).toBe(true);
  });

  it("parça açıklıktan genişse parmaklar açık kalır", () => {
    const plan = buildPlan(meta(), event("fail", "too_wide"), [part], start);
    expect(poseAt(plan, plan.attach).open).toBeGreaterThan(4.5);
  });

  it("vakumlu tutucuda parmak hareketi yoktur, parça yine taşınır", () => {
    const plan = buildPlan(meta("vacuum"), event("success", "ok", { action: { angle_idx: 0, row: 21, col: 21, theta_deg: 0 } }), [part], start, tray);
    expect(poseAt(plan, plan.attach).open).toBe(start.open);
    expect(plan.mode).toBe("success");
  });

  it("hareket süreklidir: ardışık örnekler arasında sıçrama yok", () => {
    for (const ev of [event("success", "ok"), event("fail", "slip"), event("collision", "wall", { anim: { top_z: 0, stop_z: 20 } })]) {
      const plan = buildPlan(meta(), ev, [part], start, tray);
      const ts = sample(plan, 2000);
      for (let i = 1; i < ts.length; i++) {
        const a = poseAt(plan, ts[i - 1]);
        const b = poseAt(plan, ts[i]);
        expect(Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)).toBeLessThan(1.5);
      }
    }
  });
});

describe("yardımcılar", () => {
  it("açı ara değeri 180° periyotludur (kısa yoldan döner)", () => {
    const d = (x: number) => (x * Math.PI) / 180;
    const mid = lerpAngle(d(170), d(10), 0.5);
    expect(Math.abs(Math.cos(2 * mid) - Math.cos(2 * d(0)))).toBeLessThan(1e-9); // 0° ≡ 180°
    expect(lerpAngle(d(0), d(90), 1)).toBeCloseTo(d(90), 9);
  });

  it("kapanma yönündeki genişlik", () => {
    expect(extentAlong(part, 0)).toBeCloseTo(10, 9);
    expect(extentAlong(part, Math.PI / 2)).toBeCloseTo(4, 9);
    expect(extentAlong({ ...part, kind: "disc", length: 6, width: 6 }, 1.234)).toBe(6);
  });
});
