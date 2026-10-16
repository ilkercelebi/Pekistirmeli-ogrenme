import { RoundedBox } from "@react-three/drei";
import type { Meta, Part } from "../api";
import { finishFor } from "./util";

export function PartMesh({ part, meta, local }: { part: Part; meta?: Meta; local?: boolean }) {
  const f = finishFor(part);
  const h = meta ? (meta.grid.size - 1) / 2 : 0;
  const pos: [number, number, number] = local ? [0, 0, 0] : [part.cx - h, part.base + part.thickness / 2, part.cy - h];
  const mat = <meshStandardMaterial color={f.color} metalness={f.metalness} roughness={f.roughness} envMapIntensity={1.1} />;
  if (part.kind === "disc") {
    return (
      <group position={pos}>
        <mesh castShadow receiveShadow>
          <cylinderGeometry args={[part.length / 2, part.length / 2, part.thickness, 48]} />
          {mat}
        </mesh>
        {/* üst yüzde işleme izi */}
        <mesh position={[0, part.thickness / 2 + 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[part.length * 0.16, part.length * 0.2, 40]} />
          <meshStandardMaterial color="#000" transparent opacity={0.22} />
        </mesh>
      </group>
    );
  }
  return (
    <group position={pos} rotation={[0, -part.angle, 0]}>
      <RoundedBox args={[part.length, part.thickness, part.width]} radius={Math.min(0.35, part.thickness * 0.18)} smoothness={3} castShadow receiveShadow>
        {mat}
      </RoundedBox>
    </group>
  );
}
