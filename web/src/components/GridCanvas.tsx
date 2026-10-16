// Kuşbakışı ızgara çizimleri: derinlik kamerası görüntüsü, Q-haritası, özellik haritaları.
import { useEffect, useRef, useState } from "react";
import type { GraspAction, Meta } from "../api";
import { C, heightColor, qColor, rgb } from "../theme";

type Grid = (number | null)[][];

export interface Footprint {
  action: GraspAction;
  color: string;
  dashed?: boolean;
}

interface Props {
  meta: Meta;
  mode: "depth" | "q" | "discrete";
  values: Grid;
  discreteColors?: string[];          // mode="discrete": kova indeksine göre renk
  footprints?: Footprint[];
  crop?: { row: number; col: number; radius: number };
  size?: number;                      // çizim çözünürlüğü (px)
  onHover?: (cell: { row: number; col: number } | null) => void;
  onClick?: (cell: { row: number; col: number }) => void;
  tooltip?: (row: number, col: number) => string | null;
  cursor?: string;
}

function drawFootprint(ctx: CanvasRenderingContext2D, meta: Meta, fp: Footprint, cs: number, ox: number, oy: number) {
  const g = meta.gripper;
  const th = ((fp.action.theta_deg ?? meta.angles_deg[fp.action.angle_idx] ?? 0) * Math.PI) / 180;
  const cx = (fp.action.col - ox + 0.5) * cs;
  const cy = (fp.action.row - oy + 0.5) * cs;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(th);
  ctx.lineWidth = Math.max(1.5, cs * 0.18);
  ctx.strokeStyle = fp.color;
  if (g.kind === "vacuum") {
    ctx.beginPath();
    ctx.arc(0, 0, g.cup_radius * cs, 0, Math.PI * 2);
    ctx.fillStyle = `${fp.color}40`;
    ctx.fill();
    ctx.stroke();
  } else {
    const off = g.opening / 2 + g.finger_thickness / 2;
    const L = g.opening + 2 * g.finger_thickness;
    ctx.setLineDash([cs * 0.5, cs * 0.4]);
    ctx.lineWidth = Math.max(1, cs * 0.1);
    ctx.strokeRect((-L / 2) * cs, (-g.finger_width / 2) * cs, L * cs, g.finger_width * cs);
    ctx.beginPath();
    ctx.moveTo(-off * cs, 0);
    ctx.lineTo(off * cs, 0);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = fp.dashed ? `${fp.color}88` : fp.color;
    for (const s of [-1, 1]) {
      ctx.fillRect((s * off - g.finger_thickness / 2) * cs, (-g.finger_width / 2) * cs, g.finger_thickness * cs, g.finger_width * cs);
      ctx.strokeStyle = "#0a0d11";
      ctx.lineWidth = Math.max(1, cs * 0.08);
      ctx.strokeRect((s * off - g.finger_thickness / 2) * cs, (-g.finger_width / 2) * cs, g.finger_thickness * cs, g.finger_width * cs);
    }
  }
  ctx.restore();
  // merkez artısı
  ctx.strokeStyle = fp.color;
  ctx.lineWidth = Math.max(1.5, cs * 0.16);
  const r = cs * 0.9;
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();
}

export default function GridCanvas({
  meta, mode, values, discreteColors, footprints = [], crop, size = 560, onHover, onClick, tooltip, cursor,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const N = meta.grid.size;
  const t = meta.grid.wall_thickness;
  const ox = crop ? Math.max(0, crop.col - crop.radius) : 0;
  const oy = crop ? Math.max(0, crop.row - crop.radius) : 0;
  const span = crop ? Math.min(N, 2 * crop.radius + 1) : N;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr;
    cv.height = size * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const cs = size / span;
    ctx.fillStyle = "#07090c";
    ctx.fillRect(0, 0, size, size);
    for (let r = oy; r < Math.min(N, oy + span); r++) {
      for (let c = ox; c < Math.min(N, ox + span); c++) {
        const wall = r < t || c < t || r >= N - t || c >= N - t;
        const v = values[r]?.[c];
        let col: string;
        if (wall) col = C.wall;
        else if (v == null || v < -0.5 && mode === "discrete") col = "#0d1117";
        else if (mode === "depth") col = rgb(heightColor(Math.max(0, v), meta.grid.max_stack_height));
        else if (mode === "q") col = rgb(qColor(Math.max(-1, Math.min(1, v))));
        else col = discreteColors?.[v] ?? "#333";
        ctx.fillStyle = col;
        ctx.fillRect((c - ox) * cs, (r - oy) * cs, Math.ceil(cs), Math.ceil(cs));
      }
    }
    // duvar kenarı
    ctx.strokeStyle = "rgba(255,255,255,.18)";
    ctx.lineWidth = 1;
    ctx.strokeRect((t - ox) * cs, (t - oy) * cs, (N - 2 * t) * cs, (N - 2 * t) * cs);
    for (const fp of footprints) drawFootprint(ctx, meta, fp, cs, ox, oy);
  }, [values, footprints, mode, discreteColors, size, span, ox, oy, N, t, meta]);

  const cellAt = (e: React.MouseEvent) => {
    const rect = ref.current!.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * span) + ox;
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * span) + oy;
    if (row < 0 || col < 0 || row >= N || col >= N) return null;
    return { row, col, x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  return (
    <div className="canvas-wrap">
      <canvas
        ref={ref}
        style={{ cursor: cursor ?? (onClick ? "crosshair" : "default"), aspectRatio: "1 / 1" }}
        onMouseMove={(e) => {
          const cell = cellAt(e);
          onHover?.(cell ? { row: cell.row, col: cell.col } : null);
          if (cell && tooltip) {
            const text = tooltip(cell.row, cell.col);
            setTip(text ? { x: cell.x, y: cell.y, text } : null);
          }
        }}
        onMouseLeave={() => {
          onHover?.(null);
          setTip(null);
        }}
        onClick={(e) => {
          const cell = cellAt(e);
          if (cell) onClick?.({ row: cell.row, col: cell.col });
        }}
      />
      {tip && <div className="canvas-tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </div>
  );
}
