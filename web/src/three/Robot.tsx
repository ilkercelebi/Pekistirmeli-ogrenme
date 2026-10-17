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
import { buildPlan, carriedPose, poseAt, type Plan, type Pose } from "./plan";
import { HOME } from "./util";

export const FINGER_H = 5.5;
/** Bu sonuçlarda parça animasyon boyunca robot tarafından çizilir (kıpırdar ya da taşınır). */
export const MOVING_CODES = ["ok", "slip", "covered"];
const BODY_H = 3.4;
const FLANGE_H = 1.6;
export const GANTRY = { postX: 47, postZ: 38, railY: 64, bridgeY: 67 };
const COLUMN_LEN = 34;

export function toolHeight(meta: Meta) {
  return meta.gripper.kind === "vacuum" ? 9.2 : FINGER_H + BODY_H + FLANGE_H;
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
      tRef.current += Math.min(dt, 0.1) * speed; // yavaş bilgisayarda da gerçek zamanlı kalsın
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
        carried.current.visible = !!cp;
        if (cp) {
          carried.current.position.set(cp.pos[0], cp.pos[1], cp.pos[2]);
          carried.current.rotation.y = cp.yaw;
        }
      }
      if (t >= plan.total) {
        doneRef.current = true;
        poseRef.current = plan.keys[plan.keys.length - 1].pose;
        // parçanın son pozu sahnedeki yerine (kutu ya da tepsi) birebir denk gelir
        if (carried.current) carried.current.visible = false;
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

  const carriedPart = event && event.part != null && MOVING_CODES.includes(event.code)
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
        {carriedPart && <PartMesh part={{ ...carriedPart, angle: 0 }} local />}
      </group>
    </group>
  );
}
