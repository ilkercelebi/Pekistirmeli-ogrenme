// Kavrama animasyonu planı: robotun ve parçanın hareketi her sonuç türü için fiziksel olarak tutarlı mı?
import { describe, expect, it } from "vitest";
import type { GraspEvent, Meta, Part } from "../api";
import { buildPlan, carriedPose, poseAt, type PartPose, type Plan } from "./plan";
import { HOME, extentAlong, lerpAngle } from "./util";

const meta = (kind: "parallel_jaw" | "vacuum" = "parallel_jaw"): Meta => ({
  gripper: { kind, opening: 9, finger_thickness: 1, finger_width: 3, grasp_depth: 1.5, cup_radius: 1.5 },
  grid: { size: 44, wall_thickness: 2, wall_height: 20, max_stack_height: 15 },
  angles_deg: kind === "vacuum" ? [0] : [0, 45, 90, 135],
  rewards: { success: 1, collision: -1, fail: -0.2 },
  features: [], n_states: 15552, dims: [], n_parts: [6, 14], candidates: 6400,
});

const part: Part = { id: 3, kind: "box", cx: 21, cy: 21, length: 10, width: 4, angle: 0, base: 0, thickness: 3, alive: true };
const start = { ...HOME, th: 0, open: 5 };
const tray: [number, number, number, number] = [33, 1.6, -8.2, 0.12];
const H = (44 - 1) / 2;
const home = (p: Part = part): [number, number, number] => [p.cx - H, p.base + p.thickness / 2, p.cy - H];

function event(outcome: GraspEvent["outcome"], code: string, extra: Partial<GraspEvent> = {}): GraspEvent {
  return {
    action: { angle_idx: 2, row: 21, col: 21, theta_deg: 90 },
    reward: outcome === "success" ? 1 : outcome === "collision" ? -1 : -0.2,
    outcome, code, reason: code, part: part.id, anim: { top_z: 3, stop_z: 1.5 }, ...extra,
  };
}

const sample = (plan: Plan, n = 600) => Array.from({ length: n + 1 }, (_, i) => (plan.total * i) / n);
const near = (a: number[], b: number[], eps = 1e-6) => a.every((v, i) => Math.abs(v - b[i]) < eps);
/** 180° simetrik parça için yön farkı (0 = aynı görünüm). */
const yawDiff = (a: number, b: number) => Math.abs(Math.sin(a - b));

/** Parça pozunda ani sıçrama yok: konum ve yön sürekli değişir. */
function expectContinuous(plan: Plan) {
  const ts = sample(plan, 3000);
  let prev: PartPose | null = carriedPose(plan, 0);
  for (const t of ts.slice(1)) {
    const p = carriedPose(plan, t);
    if (prev && p) {
      expect(Math.hypot(p.pos[0] - prev.pos[0], p.pos[1] - prev.pos[1], p.pos[2] - prev.pos[2])).toBeLessThan(0.5);
      expect(yawDiff(p.yaw, prev.yaw)).toBeLessThan(0.05);
    }
    prev = p;
  }
}

describe("kavrama: temas, ortalama ve hizalama", () => {
  it("parmaklar değene kadar parça kıpırdamaz", () => {
    // tutucu 4 cm'lik parçanın merkezinden 1 cm yana (x yönünde) iner ve x yönünde sıkar:
    // uzak uç tutucu merkezinden 3 cm, parmak iç yüzü 4.5 cm'den kapanmaya başlar
    const plan = buildPlan(meta(), event("success", "ok", { action: { angle_idx: 0, row: 21, col: 22, theta_deg: 0 } }),
      [{ ...part, length: 4 }], start, tray);
    expect(plan.contactInner).toBeCloseTo(3, 6);
    const contact = plan.gripStart + 0.05;
    expect(near(carriedPose(plan, plan.gripStart)!.pos, home())).toBe(true);
    expect(near(carriedPose(plan, contact)!.pos, home())).toBe(true);
    // kapandıktan sonra parça parmakların ortasına kaymıştır (tutucu 1 cm sağda)
    const s = carriedPose(plan, plan.attach)!;
    expect(s.pos[0]).toBeCloseTo(home()[0] + 1, 6);
  });

  it("parmaklar parçanın kapanma yönündeki genişliğine kadar kapanır", () => {
    const plan = buildPlan(meta(), event("success", "ok"), [part], start, tray);
    expect(poseAt(plan, plan.attach).open).toBeCloseTo(extentAlong(part, Math.PI / 2) / 2 + 0.5, 6);
  });

  it("çapraz tutulan parça parmaklara paralel olacak şekilde döner", () => {
    const plan = buildPlan(meta(), event("success", "ok", { action: { angle_idx: 1, row: 21, col: 21, theta_deg: 45 } }),
      [{ ...part, length: 8, width: 3 }], start, tray);
    const before = carriedPose(plan, plan.gripStart)!;
    const after = carriedPose(plan, plan.attach)!;
    expect(yawDiff(after.yaw, before.yaw)).toBeCloseTo(Math.sin(Math.PI / 4), 6);   // 45° döndü
    expect(poseAt(plan, plan.attach).open).toBeCloseTo(3 / 2 + 0.5, 6);              // kısa kenardan sıkıyor
  });
});

describe("taşıma ve tepsiye yerleştirme", () => {
  it("parça tepsideki yerine ve yönüne tam oturur, robot eve döner", () => {
    const plan = buildPlan(meta(), event("success", "ok"), [part], start, tray);
    expect(plan.mode).toBe("success");
    const end = carriedPose(plan, plan.total)!;
    expect(near(end.pos, tray.slice(0, 3))).toBe(true);
    expect(yawDiff(end.yaw, tray[3])).toBeLessThan(1e-9);
    const r = poseAt(plan, plan.total);
    expect([r.x, r.y, r.z]).toEqual([HOME.x, HOME.y, HOME.z]);
  });

  it("bırakmadan hemen önce parça tepsinin hemen üstündedir (düşürülmez, yerleştirilir)", () => {
    const plan = buildPlan(meta(), event("success", "ok"), [part], start, tray);
    const p = carriedPose(plan, plan.release - 1e-6)!;
    expect(Math.hypot(p.pos[0] - tray[0], p.pos[2] - tray[2])).toBeLessThan(1e-6);
    expect(p.pos[1] - tray[1]).toBeGreaterThan(0);
    expect(p.pos[1] - tray[1]).toBeLessThan(0.5);
  });

  for (const [name, ev, parts] of [
    ["düz kavrama", event("success", "ok"), [part]],
    ["çapraz kavrama", event("success", "ok", { action: { angle_idx: 1, row: 21, col: 21, theta_deg: 45 } }), [{ ...part, length: 8, width: 3 }]],
    ["kaçık kavrama", event("success", "ok", { action: { angle_idx: 2, row: 21, col: 24, theta_deg: 90 } }), [part]],
    ["disk", event("success", "ok"), [{ ...part, kind: "disc", length: 6, width: 6 } as Part]],
  ] as [string, GraspEvent, Part[]][]) {
    it(`parça hiçbir anda sıçramaz (${name})`, () => expectContinuous(buildPlan(meta(), ev, parts, start, tray)));
  }
});

describe("başarısız kavramalar", () => {
  it("kayan parça kalkar, sonra kutudaki yerine düşer", () => {
    const plan = buildPlan(meta(), event("fail", "slip"), [part], start);
    expect(plan.mode).toBe("slip");
    const peak = Math.max(...sample(plan).map((t) => carriedPose(plan, t)!.pos[1]));
    expect(peak).toBeGreaterThan(home()[1] + 1);
    const end = carriedPose(plan, plan.total)!;
    expect(near(end.pos, home())).toBe(true);
    expectContinuous(plan);
  });

  it("üstü kapalı parça çekiştirilir ama yerinden kalkmaz", () => {
    const plan = buildPlan(meta(), event("fail", "covered"), [part], start);
    expect(plan.mode).toBe("tug");
    for (const t of sample(plan)) {
      const p = carriedPose(plan, t)!;
      expect(Math.hypot(p.pos[0] - home()[0], p.pos[2] - home()[2])).toBeLessThan(1e-9);
      expect(p.pos[1] - home()[1]).toBeLessThan(0.5);
    }
    expect(near(carriedPose(plan, plan.total)!.pos, home())).toBe(true);
  });

  it("çarpışmada tutucu duvar üstünde durur, sarsılır ve parçaya dokunulmaz", () => {
    const plan = buildPlan(meta(), event("collision", "wall", { part: null, anim: { top_z: 0, stop_z: 20 } }), [part], start);
    expect(plan.shake).not.toBeNull();
    expect(Math.min(...sample(plan).map((t) => poseAt(plan, t).y))).toBeCloseTo(20, 6);
    expect(sample(plan).every((t) => carriedPose(plan, t) === null)).toBe(true);
  });

  it("boş noktada parmaklar tamamen kapanır, geniş parçada açık kalır", () => {
    const empty = buildPlan(meta(), event("fail", "empty", { part: null }), [part], start);
    expect(poseAt(empty, empty.attach).open).toBeCloseTo(0.9, 6);
    const wide = buildPlan(meta(), event("fail", "too_wide"), [part], start);
    expect(poseAt(wide, wide.attach).open).toBeGreaterThan(4.5);
    expect(carriedPose(wide, wide.total)).toBeNull();
  });

  it("vakumda parmak hareketi yoktur, parça yine tepsiye yerleşir", () => {
    const plan = buildPlan(meta("vacuum"), event("success", "ok", { action: { angle_idx: 0, row: 21, col: 21, theta_deg: 0 } }),
      [part], start, tray);
    expect(poseAt(plan, plan.attach).open).toBe(start.open);
    expect(near(carriedPose(plan, plan.total)!.pos, tray.slice(0, 3))).toBe(true);
    expectContinuous(plan);
  });
});

describe("yardımcılar", () => {
  it("açı ara değeri 180° periyotludur", () => {
    const d = (x: number) => (x * Math.PI) / 180;
    expect(Math.abs(Math.cos(2 * lerpAngle(d(170), d(10), 0.5)) - 1)).toBeLessThan(1e-9);
  });
});
