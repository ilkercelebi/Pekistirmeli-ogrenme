// Robot hücresinin sabit kısımları ve sahne bileşeni (Canvas).
import { Environment, Grid, Lightformer, Line, OrbitControls, PerspectiveCamera, RoundedBox } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import type { GraspAction, GraspEvent, Meta, Outcome, Part } from "../api";
import { C, heightColor, qColor } from "../theme";
import Robot, { FINGER_H, GANTRY, MOVING_CODES } from "./Robot";
import { PartMesh } from "./Parts";
import { TRAY, toWorld } from "./util";

// ------------------------------------------------------------------ parçalar

// ------------------------------------------------------------------ kutu, zemin, tepsi

const binMat = { color: "#2c5f96", roughness: 0.58, metalness: 0.02 };

function Bin({ meta }: { meta: Meta }) {
  const N = meta.grid.size;
  const t = meta.grid.wall_thickness;
  const H = meta.grid.wall_height;
  const outer = N / 2;
  const inner = outer - t;
  return (
    <group>
      <mesh position={[0, -0.5, 0]} receiveShadow castShadow>
        <boxGeometry args={[N, 1, N]} />
        <meshStandardMaterial color="#244f7e" roughness={0.75} />
      </mesh>
      {[-1, 1].map((s) => (
        <group key={s}>
          <RoundedBox args={[N, H, t]} radius={0.35} position={[0, H / 2, s * (inner + t / 2)]} castShadow receiveShadow>
            <meshStandardMaterial {...binMat} />
          </RoundedBox>
          <RoundedBox args={[t, H, 2 * inner]} radius={0.35} position={[s * (inner + t / 2), H / 2, 0]} castShadow receiveShadow>
            <meshStandardMaterial {...binMat} />
          </RoundedBox>
          {/* üst kenar dudağı */}
          <RoundedBox args={[N + 1.4, 1.1, t + 1]} radius={0.4} position={[0, H - 0.2, s * (inner + t / 2 + 0.35)]} castShadow>
            <meshStandardMaterial {...binMat} color="#3369a3" />
          </RoundedBox>
          <RoundedBox args={[t + 1, 1.1, N + 1.4]} radius={0.4} position={[s * (inner + t / 2 + 0.35), H - 0.2, 0]} castShadow>
            <meshStandardMaterial {...binMat} color="#3369a3" />
          </RoundedBox>
          {/* tutma yeri */}
          <mesh position={[s * (outer + 0.36), H - 4, 0]} rotation={[0, Math.PI / 2, 0]}>
            <planeGeometry args={[9, 2.2]} />
            <meshStandardMaterial color="#10253d" roughness={0.9} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function Tray() {
  const { x, z, w, d, h } = TRAY;
  const m = <meshStandardMaterial color="#59616b" roughness={0.6} metalness={0.2} />;
  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, -0.5, 0]} receiveShadow castShadow>
        <boxGeometry args={[w, 1, d]} />
        {m}
      </mesh>
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh position={[0, h / 2, (s * d) / 2]} castShadow receiveShadow>
            <boxGeometry args={[w, h, 0.8]} />
            {m}
          </mesh>
          <mesh position={[(s * w) / 2, h / 2, 0]} castShadow receiveShadow>
            <boxGeometry args={[0.8, h, d]} />
            {m}
          </mesh>
        </group>
      ))}
    </group>
  );
}

/** Tepsideki i. parçanın konumu ve dönüşü (x, y, z, rotY). */
export function traySlot(i: number, part: Part, below: number): [number, number, number, number] {
  const col = i % 3;
  const jitter = (((i * 37) % 7) - 3) * 0.25;
  return [TRAY.x + jitter, below + part.thickness / 2 + 0.05, TRAY.z + (col - 1) * 8.2, ((i * 53) % 9 - 4) * 0.04];
}

function TrayParts({ parts, order }: { parts: Part[]; order: number[] }) {
  const heights = [0, 0, 0];
  return (
    <group>
      {order.map((id, i) => {
        const p = parts.find((q) => q.id === id);
        if (!p) return null;
        const col = i % 3;
        const [x, y, z, rot] = traySlot(i, p, heights[col]);
        heights[col] += p.thickness;
        return (
          <group key={id} position={[x, y, z]} rotation={[0, rot, 0]}>
            <PartMesh part={{ ...p, angle: 0 }} local />
          </group>
        );
      })}
    </group>
  );
}

function hazardTexture() {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 32;
  const g = c.getContext("2d")!;
  g.fillStyle = "#e5b80b";
  g.fillRect(0, 0, 256, 32);
  g.fillStyle = "#121212";
  for (let x = -32; x < 256 + 32; x += 32) {
    g.beginPath();
    g.moveTo(x, 32);
    g.lineTo(x + 16, 32);
    g.lineTo(x + 32, 0);
    g.lineTo(x + 16, 0);
    g.fill();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.repeat.set(10, 1);
  return tex;
}

function Floor() {
  const tex = useMemo(hazardTexture, []);
  const S = 118;
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.02, 0]} receiveShadow>
        <planeGeometry args={[900, 900]} />
        <meshStandardMaterial color="#14171b" roughness={0.94} metalness={0.05} />
      </mesh>
      <Grid
        position={[0, -1.0, 0]} args={[400, 400]} cellSize={10} cellThickness={0.6} cellColor="#1d242c"
        sectionSize={50} sectionThickness={1} sectionColor="#27313c" fadeDistance={330} fadeStrength={1.4} infiniteGrid
      />
      {[0, 1, 2, 3].map((i) => (
        <group key={i} rotation={[0, (i * Math.PI) / 2, 0]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.98, S / 2]}>
            <planeGeometry args={[S + 3, 3]} />
            <meshStandardMaterial map={tex} roughness={0.8} polygonOffset polygonOffsetFactor={-1} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

// ------------------------------------------------------------------ gantry iskeleti, kamera, sinyal kulesi

const profile = { color: "#a6aeb6", metalness: 0.85, roughness: 0.36 };

function GantryFrame() {
  const { postX, postZ, railY } = GANTRY;
  return (
    <group>
      {[-1, 1].map((sx) =>
        [-1, 1].map((sz) => (
          <group key={`${sx}${sz}`} position={[sx * postX, 0, sz * postZ]}>
            <mesh position={[0, railY / 2 - 0.5, 0]} castShadow receiveShadow>
              <boxGeometry args={[3.4, railY + 1, 3.4]} />
              <meshStandardMaterial {...profile} />
            </mesh>
            <mesh position={[0, -0.7, 0]} receiveShadow>
              <boxGeometry args={[7, 0.6, 7]} />
              <meshStandardMaterial color="#3a4048" metalness={0.6} roughness={0.5} />
            </mesh>
          </group>
        )),
      )}
      {[-1, 1].map((sx) => (
        <group key={sx}>
          <mesh position={[sx * postX, railY + 0.9, 0]} castShadow receiveShadow>
            <boxGeometry args={[3.6, 1.8, postZ * 2 + 3.4]} />
            <meshStandardMaterial {...profile} />
          </mesh>
          <mesh position={[sx * postX, railY + 2.05, 0]}>
            <boxGeometry args={[1.4, 0.5, postZ * 2]} />
            <meshStandardMaterial color="#d6dbe0" metalness={0.95} roughness={0.18} />
          </mesh>
          {/* kablo kanalı */}
          <mesh position={[sx * (postX + 2.6), railY - 3, 0]} castShadow>
            <boxGeometry args={[1.6, 2.4, postZ * 2]} />
            <meshStandardMaterial color="#1b1e22" roughness={0.8} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function DepthCamera({ meta, pulse }: { meta: Meta; pulse: number }) {
  const lens = useRef<THREE.MeshStandardMaterial>(null);
  const start = useRef(0);
  useEffect(() => {
    start.current = performance.now();
  }, [pulse]);
  useFrame(() => {
    if (!lens.current) return;
    const k = Math.max(0, 1 - (performance.now() - start.current) / 700);
    lens.current.emissiveIntensity = 0.6 + k * 4;
  });
  const y = 88;
  const H = meta.grid.wall_height;
  const inner = meta.grid.size / 2 - meta.grid.wall_thickness;
  const armX = -GANTRY.postX - 7;
  return (
    <group>
      <mesh position={[armX, (y + 3) / 2, 0]} castShadow>
        <boxGeometry args={[3, y + 3, 3]} />
        <meshStandardMaterial {...profile} />
      </mesh>
      <mesh position={[armX / 2, y + 2.4, 0]} castShadow>
        <boxGeometry args={[-armX + 3, 2.4, 2.4]} />
        <meshStandardMaterial {...profile} />
      </mesh>
      <RoundedBox args={[11, 3, 3.6]} radius={0.5} position={[0, y, 0]} castShadow>
        <meshStandardMaterial color="#1f2328" metalness={0.4} roughness={0.45} />
      </RoundedBox>
      {[-3, 0, 3].map((x, i) => (
        <mesh key={x} position={[x, y - 1.55, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[i === 1 ? 0.9 : 0.6, i === 1 ? 0.9 : 0.6, 0.3, 24]} />
          <meshStandardMaterial ref={i === 1 ? lens : undefined} color="#0b1a22" emissive="#38d6ff"
            emissiveIntensity={i === 1 ? 0.6 : 0.15} metalness={0.2} roughness={0.15} />
        </mesh>
      ))}
      {[[-1, -1], [-1, 1], [1, 1], [1, -1]].map(([sx, sz], i) => (
        <Line key={i} points={[[0, y - 1.7, 0], [sx * inner, H, sz * inner]]} color="#38d6ff" lineWidth={1} transparent opacity={0.22} />
      ))}
      <Line points={[[-inner, H, -inner], [inner, H, -inner], [inner, H, inner], [-inner, H, inner], [-inner, H, -inner]]}
        color="#38d6ff" lineWidth={1} transparent opacity={0.25} />
    </group>
  );
}

function SignalTower({ signal }: { signal: "idle" | "busy" | Outcome }) {
  const lights: [string, boolean][] = [
    [C.collision, signal === "collision"],
    [C.fail, signal === "fail" || signal === "busy" || signal === "stop"],
    [C.success, signal === "success" || signal === "idle"],
  ];
  return (
    <group position={[GANTRY.postX, GANTRY.railY + 1.8, -GANTRY.postZ]}>
      <mesh position={[0, 1, 0]} castShadow>
        <cylinderGeometry args={[1.1, 1.3, 2, 24]} />
        <meshStandardMaterial color="#26292e" metalness={0.5} roughness={0.5} />
      </mesh>
      {lights.map(([color, on], i) => (
        <mesh key={i} position={[0, 7.4 - i * 2.1, 0]}>
          <cylinderGeometry args={[1.05, 1.05, 1.9, 24]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={on ? 2.6 : 0.05} transparent opacity={on ? 1 : 0.45}
            toneMapped={!on} roughness={0.3} />
        </mesh>
      ))}
      <mesh position={[0, 8.6, 0]}>
        <cylinderGeometry args={[0.9, 1.05, 0.4, 24]} />
        <meshStandardMaterial color="#26292e" />
      </mesh>
    </group>
  );
}

// ------------------------------------------------------------------ katmanlar: Q-haritası, önizleme, tarama

function OverlayGrid({ meta, depth, values, kind }: {
  meta: Meta; depth: (number | null)[][]; values: (number | null)[][]; kind: "q" | "depth";
}) {
  const geom = useMemo(() => {
    const N = meta.grid.size;
    const t = meta.grid.wall_thickness;
    const pos: number[] = [];
    const col: number[] = [];
    const idx: number[] = [];
    let n = 0;
    for (let r = t; r < N - t; r++) {
      for (let c = t; c < N - t; c++) {
        const v = values[r]?.[c];
        if (v == null) continue;
        const [x, z] = toWorld(meta, r, c);
        const y = Math.max(0, depth[r]?.[c] ?? 0) + 0.14;
        const rgb = kind === "q" ? qColor(Math.max(-1, Math.min(1, v))) : heightColor(v, meta.grid.max_stack_height);
        for (const [dx, dz] of [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) {
          pos.push(x + dx, y, z + dz);
          col.push(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
        }
        idx.push(n, n + 2, n + 1, n, n + 3, n + 2);
        n += 4;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    return g;
  }, [meta, depth, values, kind]);
  useEffect(() => () => geom.dispose(), [geom]);
  return (
    <mesh geometry={geom} renderOrder={2}>
      <meshBasicMaterial vertexColors transparent opacity={0.84} depthWrite={false} toneMapped={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

function Ghost({ meta, action, depth, color }: { meta: Meta; action: GraspAction; depth: (number | null)[][]; color: string }) {
  const g = meta.gripper;
  const [x, z] = toWorld(meta, action.row, action.col);
  const top = Math.max(0, depth[action.row]?.[action.col] ?? 0);
  const th = ((action.theta_deg ?? meta.angles_deg[action.angle_idx]) * Math.PI) / 180;
  const off = g.opening / 2 + g.finger_thickness / 2;
  const mat = <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} transparent opacity={0.55} depthWrite={false} />;
  return (
    <group position={[x, top, z]} rotation={[0, -th, 0]}>
      {g.kind === "vacuum" ? (
        <mesh position={[0, 0.6, 0]}>
          <cylinderGeometry args={[g.cup_radius, g.cup_radius, 1.2, 32]} />
          {mat}
        </mesh>
      ) : (
        [-1, 1].map((s) => (
          <mesh key={s} position={[s * off, FINGER_H / 2, 0]}>
            <boxGeometry args={[g.finger_thickness, FINGER_H, g.finger_width]} />
            {mat}
          </mesh>
        ))
      )}
      <mesh position={[0, 14, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 28, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

function ScanSweep({ meta, trigger }: { meta: Meta; trigger: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const start = useRef(-1e9);
  useEffect(() => {
    if (trigger > 0) start.current = performance.now();
  }, [trigger]);
  const inner = meta.grid.size / 2 - meta.grid.wall_thickness;
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const f = (performance.now() - start.current) / 900;
    m.visible = f >= 0 && f <= 1;
    if (m.visible) m.position.z = -inner + 2 * inner * f;
  });
  return (
    <mesh ref={ref} position={[0, meta.grid.wall_height / 2, 0]} visible={false}>
      <planeGeometry args={[2 * inner, meta.grid.wall_height]} />
      <meshBasicMaterial color="#38d6ff" transparent opacity={0.13} side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
    </mesh>
  );
}

// ------------------------------------------------------------------ sahne

export interface RobotCellProps {
  meta: Meta;
  parts: Part[];
  depth?: (number | null)[][];
  event?: GraspEvent | null;
  eventKey?: number;
  onEventDone?: () => void;
  speed?: number;
  overlay?: { kind: "q" | "depth"; values: (number | null)[][] } | null;
  preview?: { action: GraspAction; color: string } | null;
  height?: number | string;
  signal?: "idle" | "busy" | Outcome;
  children?: ReactNode;
  view?: "iso" | "top" | "close";
  /** Kameranın odaklanacağı dünya noktası (x, z); verilirse "close" görünüm bu noktaya bakar. */
  focus?: [number, number] | null;
}

const VIEWS = {
  iso: { pos: [122, 112, 146] as [number, number, number], target: [2, 18, 0] as [number, number, number], fov: 30 },
  close: { pos: [48, 82, 64] as [number, number, number], target: [0, 4, 2] as [number, number, number], fov: 34 },
  top: { pos: [0.01, 150, 0.01] as [number, number, number], target: [0, 0, 0] as [number, number, number], fov: 30 },
};

export default function RobotCell({
  meta, parts, depth, event = null, eventKey = 0, onEventDone, speed = 1, overlay, preview, height = 600,
  signal = "idle", children, view = "iso", focus = null,
}: RobotCellProps) {
  // Tepsideki parçaların sırası: kutudan alınan parçalar sırayla tepsiye dizilir.
  const orderRef = useRef<number[]>([]);
  const dead = parts.filter((p) => !p.alive).map((p) => p.id);
  const kept = orderRef.current.filter((id) => dead.includes(id));
  for (const id of dead) if (!kept.includes(id)) kept.push(id);
  orderRef.current = kept;

  const animating = !!event;
  const hidden = animating && event && MOVING_CODES.includes(event.code) ? event.part : null;
  const alive = parts.filter((p) => p.alive && p.id !== hidden);

  let trayTarget: [number, number, number, number] | undefined;
  if (event && event.code === "ok" && event.part != null) {
    const p = parts.find((q) => q.id === event.part);
    if (p) {
      const i = kept.length;
      let below = 0;
      kept.forEach((id, j) => {
        if (j % 3 === i % 3) below += parts.find((q) => q.id === id)?.thickness ?? 0;
      });
      trayTarget = traySlot(i, p, below);
    }
  }

  const v = view === "close" && focus
    ? { pos: [focus[0] + 34, 64, focus[1] + 58] as [number, number, number], target: [focus[0] * 0.6, 16, focus[1] * 0.6] as [number, number, number], fov: 34 }
    : VIEWS[view];
  const camKey = `${view}-${focus ? focus.join(",") : ""}`;
  const scanKey = useMemo(() => Math.random(), [parts]); // her yeni durumda kamera taraması

  return (
    <div className="viewport" style={{ height }}>
      <div className="vp-loading">3D sahne hazırlanıyor…</div>
      <Canvas shadows dpr={[1, 2]} gl={{ antialias: true, toneMappingExposure: 1.3 }}>
        <PerspectiveCamera key={camKey} makeDefault position={v.pos} fov={v.fov} near={1} far={2000} />
        <OrbitControls key={`c${camKey}`} target={v.target} enablePan={false} minDistance={45} maxDistance={260}
          maxPolarAngle={1.42} enableDamping dampingFactor={0.08} />
        <color attach="background" args={["#090c10"]} />
        <fog attach="fog" args={["#090c10", 220, 520]} />

        <hemisphereLight args={["#c4d8ff", "#2a2016", 0.7]} />
        <directionalLight position={[55, 120, 45]} intensity={3} castShadow shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-80} shadow-camera-right={80} shadow-camera-top={80} shadow-camera-bottom={-80}
          shadow-camera-near={10} shadow-camera-far={320} shadow-bias={-0.0004} shadow-normalBias={0.02} />
        <directionalLight position={[-70, 60, -40]} intensity={0.55} color="#9fc2ff" />
        <spotLight position={[0, 130, 0]} angle={0.42} penumbra={0.7} intensity={2200} decay={1.7} color="#fff3e2" />

        <Environment resolution={256} frames={1}>
          <color attach="background" args={["#20262e"]} />
          <Lightformer form="rect" intensity={5} position={[0, 60, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[100, 22, 1]} />
          <Lightformer form="rect" intensity={3} position={[0, 60, -45]} rotation={[Math.PI / 2, 0, 0]} scale={[100, 12, 1]} />
          <Lightformer form="rect" intensity={3} position={[0, 60, 45]} rotation={[Math.PI / 2, 0, 0]} scale={[100, 12, 1]} />
          <Lightformer form="rect" intensity={2} color="#ffc290" position={[-90, 25, 0]} rotation={[0, Math.PI / 2, 0]} scale={[80, 40, 1]} />
          <Lightformer form="rect" intensity={1.6} color="#a8c8ff" position={[90, 25, 0]} rotation={[0, -Math.PI / 2, 0]} scale={[80, 40, 1]} />
          <Lightformer form="ring" intensity={2} position={[0, 30, 90]} scale={30} />
        </Environment>

        <Floor />
        <Bin meta={meta} />
        <Tray />
        <GantryFrame />
        <DepthCamera meta={meta} pulse={scanKey} />
        <SignalTower signal={animating ? "busy" : signal} />

        {alive.map((p) => <PartMesh key={p.id} part={p} meta={meta} />)}
        <TrayParts parts={parts} order={kept} />

        <Robot meta={meta} parts={parts} event={event} eventKey={eventKey} speed={speed} trayTarget={trayTarget} onDone={onEventDone} />

        {overlay && depth && !animating && <OverlayGrid meta={meta} depth={depth} values={overlay.values} kind={overlay.kind} />}
        {preview && depth && !animating && <Ghost meta={meta} action={preview.action} depth={depth} color={preview.color} />}
        <ScanSweep meta={meta} trigger={scanKey} />
      </Canvas>
      {children}
    </div>
  );
}
