// Kartezyen (gantry) robot, tutucu ve kavrama animasyonu.
//
// Animasyon, olayın (GraspEvent) sonucuna göre kurulan bir anahtar kare planıdır:
// hedefin üstüne git → in → kavra → kaldır → (başarılıysa) tepsiye bırak → eve dön.
// Tüm pozlar zamanın saf fonksiyonudur; böylece hız değiştirilse de animasyon tutarlı kalır.
import { RoundedBox } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { GraspEvent, Meta, Part } from "../api";
import { C, OUTCOME } from "../theme";
import { PartMesh } from "./Parts";
import { HOME, ease, extentAlong, lerp, lerpAngle, toWorld } from "./util";

type Pose = { x: number; y: number; z: number; th: number; open: number };
type Key = { t: number; pose: Pose; ease?: (t: number) => number };

interface Plan {
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

export const FINGER_H = 5.5;
const BODY_H = 3.4;
const FLANGE_H = 1.6;
export const GANTRY = { postX: 47, postZ: 38, railY: 64, bridgeY: 67 };
const COLUMN_LEN = 34;

export function toolHeight(meta: Meta) {
  return meta.gripper.kind === "vacuum" ? 9.2 : FINGER_H + BODY_H + FLANGE_H;
}

function poseAt(plan: Plan, t: number): Pose {
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

function buildPlan(meta: Meta, ev: GraspEvent, parts: Part[], start: Pose, trayTarget?: [number, number, number, number]): Plan {
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
      push(0.6, { y: HOME.y, z: HOME.z });
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
function carriedPose(plan: Plan, t: number): [number, number, number] | null {
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

const steel = { color: "#c3cad1", metalness: 0.92, roughness: 0.28 };
const alu = { color: "#9aa3ac", metalness: 0.85, roughness: 0.38 };

interface RobotProps {
  meta: Meta;
  parts: Part[];
  event: GraspEvent | null;
  eventKey: number;
  speed: number;
  trayTarget?: [number, number, number, number];
  onDone?: () => void;
}

export default function Robot({ meta, parts, event, eventKey, speed, trayTarget, onDone }: RobotProps) {
  const g = meta.gripper;
  const open0 = g.opening / 2 + g.finger_thickness / 2;
  const toolH = toolHeight(meta);
  const poseRef = useRef<Pose>({ ...HOME, th: 0, open: open0 });
  const planRef = useRef<Plan | null>(null);
  const tRef = useRef(0);
  const doneRef = useRef(true);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  const bridge = useRef<THREE.Group>(null);
  const carriage = useRef<THREE.Group>(null);
  const column = useRef<THREE.Mesh>(null);
  const tool = useRef<THREE.Group>(null);
  const fingerA = useRef<THREE.Mesh>(null);
  const fingerB = useRef<THREE.Mesh>(null);
  const led = useRef<THREE.MeshStandardMaterial>(null);
  const flash = useRef<THREE.PointLight>(null);
  const carried = useRef<THREE.Group>(null);

  useEffect(() => {
    if (!event) return;
    planRef.current = buildPlan(meta, event, parts, { ...poseRef.current }, trayTarget);
    tRef.current = 0;
    doneRef.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eventKey]);

  const ledColor = useMemo(() => new THREE.Color(C.accent), []);

  useFrame((_, dt) => {
    const plan = planRef.current;
    let pose = poseRef.current;
    let shake = 0;
    if (plan && !doneRef.current) {
      tRef.current += Math.min(dt, 0.05) * speed;
      const t = tRef.current;
      pose = poseAt(plan, t);
      if (plan.shake && t >= plan.shake[0] && t <= plan.shake[1]) {
        const f = (t - plan.shake[0]) / (plan.shake[1] - plan.shake[0]);
        shake = Math.sin(t * 70) * 0.45 * (1 - f);
      }
      const outcome = event?.outcome;
      const lit = t >= plan.outcomeAt && outcome;
      ledColor.set(lit ? OUTCOME[outcome].color : C.accent);
      if (flash.current) {
        const on = outcome === "collision" && plan.shake && t >= plan.shake[0] && t <= plan.shake[1] + 0.2;
        flash.current.intensity = on ? 900 * (0.6 + 0.4 * Math.sin(t * 40)) : 0;
        flash.current.position.set(pose.x, pose.y + 1, pose.z);
      }
      const cp = carriedPose(plan, t);
      if (carried.current) {
        carried.current.visible = !!cp || (plan.mode !== "none" && t < plan.attach);
        if (cp) carried.current.position.set(cp[0], cp[1], cp[2]);
        else if (plan.part) {
          const h = (meta.grid.size - 1) / 2;
          carried.current.position.set(plan.part.cx - h, plan.part.base + plan.part.thickness / 2, plan.part.cy - h);
        }
      }
      if (t >= plan.total) {
        doneRef.current = true;
        poseRef.current = plan.keys[plan.keys.length - 1].pose;
        // kayan parça yerinde görünmeye devam etsin (yeni durum gelene kadar)
        if (carried.current) carried.current.visible = plan.mode === "slip";
        ledColor.set(outcome ? OUTCOME[outcome].color : C.accent);
        onDoneRef.current?.();
      } else {
        poseRef.current = pose;
      }
    }
    if (led.current) {
      led.current.color.copy(ledColor);
      led.current.emissive.copy(ledColor);
    }

    const x = pose.x + shake;
    bridge.current?.position.set(0, 0, pose.z);
    carriage.current?.position.set(x, 0, 0);
    column.current?.position.set(x, pose.y + toolH + COLUMN_LEN / 2, pose.z);
    if (tool.current) {
      tool.current.position.set(x, pose.y, pose.z);
      tool.current.rotation.y = -pose.th;
    }
    fingerA.current?.position.set(-pose.open, FINGER_H / 2, 0);
    fingerB.current?.position.set(pose.open, FINGER_H / 2, 0);
  });

  const carriedPart = event && event.part != null && (event.code === "ok" || event.code === "slip")
    ? parts.find((p) => p.id === event.part)
    : undefined;
  const bodyL = g.opening + 2 * g.finger_thickness + 2.4;

  return (
    <group>
      {/* köprü: raylar üzerinde z ekseninde kayar */}
      <group ref={bridge}>
        <RoundedBox args={[GANTRY.postX * 2 + 4, 4.5, 5]} radius={0.5} position={[0, GANTRY.bridgeY, 0]} castShadow receiveShadow>
          <meshStandardMaterial color="#e2711f" metalness={0.35} roughness={0.42} />
        </RoundedBox>
        <mesh position={[0, GANTRY.bridgeY - 2.6, 2.6]}>
          <boxGeometry args={[GANTRY.postX * 2, 0.6, 0.6]} />
          <meshStandardMaterial {...steel} />
        </mesh>
        {/* araba: köprü üzerinde x ekseninde kayar */}
        <group ref={carriage}>
          <RoundedBox args={[9, 7.5, 8]} radius={0.6} position={[0, GANTRY.bridgeY, 3.2]} castShadow>
            <meshStandardMaterial color="#2b3139" metalness={0.6} roughness={0.4} />
          </RoundedBox>
          <mesh position={[0, GANTRY.bridgeY + 4.2, 3.2]} castShadow>
            <cylinderGeometry args={[2.6, 2.6, 4, 28]} />
            <meshStandardMaterial color="#1d2127" metalness={0.5} roughness={0.5} />
          </mesh>
        </group>
      </group>

      {/* dikey (Z) eksen profili */}
      <mesh ref={column} castShadow>
        <boxGeometry args={[2.8, COLUMN_LEN, 2.8]} />
        <meshStandardMaterial {...alu} />
      </mesh>

      {/* tutucu */}
      <group ref={tool}>
        {g.kind === "vacuum" ? (
          <group>
            <mesh position={[0, 0.6, 0]} castShadow>
              <cylinderGeometry args={[g.cup_radius * 0.62, g.cup_radius, 1.2, 36]} />
              <meshStandardMaterial color="#141516" roughness={0.9} metalness={0} />
            </mesh>
            {[1.6, 2.3, 3.0].map((y) => (
              <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
                <torusGeometry args={[g.cup_radius * 0.5, 0.32, 12, 28]} />
                <meshStandardMaterial color="#1c1d1f" roughness={0.85} />
              </mesh>
            ))}
            <mesh position={[0, 5.2, 0]} castShadow>
              <cylinderGeometry args={[0.55, 0.55, 3.6, 16]} />
              <meshStandardMaterial {...steel} />
            </mesh>
            <RoundedBox args={[4.4, 2.4, 4.4]} radius={0.4} position={[0, 7.8, 0]} castShadow>
              <meshStandardMaterial color="#2b3139" metalness={0.6} roughness={0.4} />
            </RoundedBox>
            <mesh position={[2.3, 8.2, 0]}>
              <sphereGeometry args={[0.45, 16, 16]} />
              <meshStandardMaterial ref={led} color={C.accent} emissive={C.accent} emissiveIntensity={2.2} toneMapped={false} />
            </mesh>
          </group>
        ) : (
          <group>
            <mesh ref={fingerA} castShadow>
              <boxGeometry args={[g.finger_thickness, FINGER_H, g.finger_width]} />
              <meshStandardMaterial {...steel} />
            </mesh>
            <mesh ref={fingerB} castShadow>
              <boxGeometry args={[g.finger_thickness, FINGER_H, g.finger_width]} />
              <meshStandardMaterial {...steel} />
            </mesh>
            {/* parmak kızağı */}
            <mesh position={[0, FINGER_H + 0.35, 0]} castShadow>
              <boxGeometry args={[bodyL - 0.6, 0.7, g.finger_width * 0.55]} />
              <meshStandardMaterial {...alu} />
            </mesh>
            <RoundedBox args={[bodyL, BODY_H, g.finger_width + 1.6]} radius={0.45} position={[0, FINGER_H + 0.7 + BODY_H / 2, 0]} castShadow>
              <meshStandardMaterial color="#2b3139" metalness={0.55} roughness={0.42} />
            </RoundedBox>
            <mesh position={[0, FINGER_H + 0.7 + BODY_H / 2, (g.finger_width + 1.6) / 2 + 0.02]}>
              <planeGeometry args={[bodyL - 1.2, 0.5]} />
              <meshStandardMaterial color={C.accent} emissive={C.accent} emissiveIntensity={0.4} />
            </mesh>
            <mesh position={[bodyL / 2 - 0.9, FINGER_H + 0.7 + BODY_H + 0.05, 0]}>
              <sphereGeometry args={[0.42, 16, 16]} />
              <meshStandardMaterial ref={led} color={C.accent} emissive={C.accent} emissiveIntensity={2.2} toneMapped={false} />
            </mesh>
            <mesh position={[0, FINGER_H + 0.7 + BODY_H + FLANGE_H / 2, 0]} castShadow>
              <cylinderGeometry args={[2.2, 2.4, FLANGE_H, 32]} />
              <meshStandardMaterial {...alu} />
            </mesh>
          </group>
        )}
      </group>

      <pointLight ref={flash} color={C.collision} intensity={0} distance={40} decay={2} />

      {/* taşınan parça (yalnızca animasyon sırasında görünür) */}
      <group ref={carried} visible={false}>
        {carriedPart && <PartMesh part={carriedPart} local />}
      </group>
    </group>
  );
}
