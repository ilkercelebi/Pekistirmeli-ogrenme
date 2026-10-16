// Python API ile konuşan ince katman ve paylaşılan tipler.

export type Outcome = "success" | "collision" | "fail";
export type GripperKind = "parallel_jaw" | "vacuum";
export type Tone = "good" | "mid" | "bad";

export interface Part {
  id: number;
  kind: "box" | "disc";
  cx: number;
  cy: number;
  length: number;
  width: number;
  angle: number;
  base: number;
  thickness: number;
  alive: boolean;
}

export interface GraspAction {
  angle_idx: number;
  row: number;
  col: number;
  theta_deg?: number;
}

export interface FeatureValue {
  name: string;
  bin?: number;
  label: string;
  tone: Tone;
}

export interface GraspEvent {
  step?: number;
  action: GraspAction;
  reward: number;
  outcome: Outcome;
  reason: string;
  code: string;
  part: number | null;
  anim: { top_z: number; stop_z: number };
  chosen_by?: "agent" | "user";
  q?: number | null;
  features?: FeatureValue[];
}

export interface SessionState {
  id: string;
  method: string;
  method_label: string;
  seed: number;
  gripper: GripperKind;
  parts: Part[];
  depth: (number | null)[][];
  parts_left: number;
  n_initial: number;
  steps: number;
  max_steps: number;
  done: boolean;
  history: GraspEvent[];
  learns: boolean;
  qmap: { q: (number | null)[][]; best_angle: number[][] } | null;
}

export interface Meta {
  gripper: {
    kind: GripperKind;
    opening: number;
    finger_thickness: number;
    finger_width: number;
    grasp_depth: number;
    cup_radius: number;
  };
  grid: { size: number; wall_thickness: number; wall_height: number; max_stack_height: number };
  angles_deg: number[];
  rewards: { success: number; collision: number; fail: number };
  features: { name: string; bins: { label: string; tone: Tone }[]; info: string }[];
  n_states: number;
  dims: number[];
  n_parts: [number, number];
  candidates: number;
}

export interface MethodInfo {
  id: string;
  label: string;
  learns: boolean;
}

export interface Scenario {
  key: string;
  title: string;
  explanation: string;
  gripper: GripperKind;
  parts: Part[];
  depth: (number | null)[][];
  event: GraspEvent;
}

export interface Preview {
  state_id: number;
  valid: boolean;
  features: FeatureValue[];
  q: number | null;
  wall_hit: boolean;
}

export interface EvalResult {
  method: string;
  label: string;
  return: number;
  success_rate: number;
  collision_rate: number;
  fail_rate: number;
  cleared: number;
  steps: number;
  [k: string]: number | string;
}

export interface CurvePoint {
  episode: number;
  epsilon: number;
  return: number;
  success_rate: number;
  collision_rate: number;
  cleared: number;
}

export interface Job {
  id: string;
  name: string;
  algo: string;
  gripper: GripperKind;
  episodes: number;
  status: "running" | "done" | "cancelled" | "error";
  episode: number;
  curve: CurvePoint[];
  evals: { episode: number; success_rate: number; collision_rate: number; cleared: number }[];
  error: string | null;
  elapsed: number;
}

export interface RunSummary {
  name: string;
  algo: string;
  gripper: GripperKind;
  episodes: number;
  last_eval: Record<string, number> | null;
}

export interface StateDesc {
  state_id: number;
  q: number;
  visits: number;
  features: FeatureValue[];
}

export interface RunDetail {
  name: string;
  curve: CurvePoint[];
  evals: { episode: number; success_rate: number; collision_rate: number; cleared: number; return: number }[];
  marginal: Record<string, (number | null)[]>;
  visited: number;
  n_states: number;
  best: StateDesc[];
  worst: StateDesc[];
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try {
      msg = (await res.json()).detail ?? msg;
    } catch {
      /* gövde JSON değil */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body: unknown) => call<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  meta: (gripper: GripperKind) => call<Meta>(`/meta?gripper=${gripper}`),
  methods: (gripper: GripperKind) => call<MethodInfo[]>(`/methods?gripper=${gripper}`),
  createSession: (gripper: GripperKind, method: string, seed: number) =>
    post<SessionState>("/session", { gripper, method, seed }),
  reset: (id: string, seed?: number) => post<SessionState>(`/session/${id}/reset`, { seed }),
  step: (id: string, action?: GraspAction) =>
    post<{ event: GraspEvent; parts_before: Part[]; state: SessionState }>(`/session/${id}/step`, {
      action: action ? { angle_idx: action.angle_idx, row: action.row, col: action.col } : null,
    }),
  preview: (id: string, a: GraspAction) => post<Preview>(`/session/${id}/preview`, a),
  featureMaps: (id: string, angle: number) => call<Record<string, number[][]>>(`/session/${id}/features?angle_idx=${angle}`),
  scenarios: () => call<Scenario[]>("/scenarios"),
  evaluate: (gripper: GripperKind, methods: string[], episodes: number) =>
    post<{ episodes: number; results: EvalResult[] }>("/evaluate", { gripper, methods, episodes }),
  startTrain: (algo: string, gripper: GripperKind, episodes: number, name?: string) =>
    post<Job>("/train", { algo, gripper, episodes, name }),
  job: (id: string) => call<Job>(`/train/${id}`),
  currentJob: () => call<Job | null>("/train"),
  cancelJob: (id: string) => post<{ ok: boolean }>(`/train/${id}/cancel`, {}),
  runs: (gripper?: GripperKind) => call<RunSummary[]>(`/runs${gripper ? `?gripper=${gripper}` : ""}`),
  run: (name: string, window = 50) => call<RunDetail>(`/runs/${encodeURIComponent(name)}?window=${window}`),
};
