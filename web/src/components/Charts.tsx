import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { C } from "../theme";

export const SERIES = [C.accent, C.info, "#bc8cff", "#39c5cf", "#f778ba", "#a5d6ff"];
export const METHOD_COLOR: Record<string, string> = { random: "#6e7681", heuristic: C.info };

const axis = { stroke: C.border, tick: { fill: C.muted, fontSize: 12 }, tickLine: false };
const tooltip = {
  contentStyle: { background: C.panel2, border: `1px solid ${C.border}`, borderRadius: 10, fontSize: 13 },
  labelStyle: { color: C.muted },
  itemStyle: { color: C.text },
};

export function LineSeries({
  data, x, lines, height = 300, yDomain, refs = [], percent,
}: {
  data: Record<string, number>[];
  x: string;
  lines: { key: string; name: string; color: string; dashed?: boolean; dot?: boolean }[];
  height?: number;
  yDomain?: [number, number];
  refs?: { y: number; label: string; color: string }[];
  percent?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 4, left: -6 }}>
        <CartesianGrid stroke={C.border} strokeDasharray="3 4" vertical={false} />
        <XAxis dataKey={x} {...axis} />
        <YAxis {...axis} domain={yDomain ?? ["auto", "auto"]} tickFormatter={percent ? (v) => `%${Math.round(v * 100)}` : undefined} />
        <Tooltip {...tooltip} formatter={(v: number) => (percent ? `%${(v * 100).toFixed(1)}` : v.toFixed(3))} />
        <Legend wrapperStyle={{ fontSize: 12, color: C.muted }} />
        {refs.map((r) => (
          <ReferenceLine key={r.label} y={r.y} stroke={r.color} strokeDasharray="5 5"
            label={{ value: r.label, fill: r.color, fontSize: 11, position: "insideTopRight" }} />
        ))}
        {lines.map((l) => (
          <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={2.2}
            dot={l.dot ? { r: 3 } : false} strokeDasharray={l.dashed ? "5 4" : undefined} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function GroupedBars({
  data, groups, height = 340,
}: { data: Record<string, number | string>[]; groups: { key: string; name: string; color: string }[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 22, right: 16, bottom: 4, left: -6 }} barCategoryGap="22%">
        <CartesianGrid stroke={C.border} strokeDasharray="3 4" vertical={false} />
        <XAxis dataKey="metric" {...axis} />
        <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => `%${Math.round(v * 100)}`} />
        <Tooltip {...tooltip} cursor={{ fill: "rgba(255,255,255,.03)" }} formatter={(v: number) => `%${(v * 100).toFixed(1)}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {groups.map((g) => (
          <Bar key={g.key} dataKey={g.key} name={g.name} fill={g.color} radius={[5, 5, 0, 0]} isAnimationActive={false}>
            <LabelList dataKey={g.key} position="top" formatter={(v: number) => `${Math.round(v * 100)}`} fill={C.muted} fontSize={10.5} />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SignedBars({ data, height = 170 }: { data: { label: string; value: number | null }[]; height?: number }) {
  const rows = data.map((d) => ({ ...d, value: d.value ?? 0, missing: d.value == null }));
  // γ > 0 olduğundan Q değerleri ±1'i aşabilir; eksen veriye göre genişler
  const lim = Math.max(1, Math.ceil(Math.max(...rows.map((r) => Math.abs(r.value))) * 2) / 2);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 18, right: 6, bottom: 0, left: -26 }}>
        <CartesianGrid stroke={C.border} strokeDasharray="3 4" vertical={false} />
        <XAxis dataKey="label" {...axis} tick={{ fill: C.muted, fontSize: 11 }} interval={0} />
        <YAxis {...axis} domain={[-lim, lim]} ticks={[-lim, 0, lim]} />
        <ReferenceLine y={0} stroke={C.muted} />
        <Tooltip {...tooltip} cursor={{ fill: "rgba(255,255,255,.03)" }} formatter={(v: number) => v.toFixed(3)} />
        <Bar dataKey="value" name="ortalama Q" radius={[4, 4, 4, 4]} isAnimationActive={false}>
          {rows.map((r, i) => <Cell key={i} fill={r.missing ? C.border : r.value >= 0 ? C.success : C.collision} />)}
          <LabelList dataKey="value" position="top" formatter={(v: number) => (v >= 0 ? "+" : "") + v.toFixed(2)} fill={C.muted} fontSize={10.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
