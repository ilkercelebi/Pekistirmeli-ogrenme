import { useCallback, useEffect, useRef, useState } from "react";
import { api, type GraspAction, type GraspEvent, type GripperKind, type Meta, type MethodInfo, type Part, type SessionState } from "./api";

const metaCache = new Map<GripperKind, Promise<Meta>>();

export function useMeta(gripper: GripperKind) {
  const [meta, setMeta] = useState<Meta | null>(null);
  useEffect(() => {
    let live = true;
    if (!metaCache.has(gripper)) metaCache.set(gripper, api.meta(gripper));
    metaCache.get(gripper)!.then((m) => live && setMeta(m)).catch(() => metaCache.delete(gripper));
    return () => {
      live = false;
    };
  }, [gripper]);
  return meta;
}

export function useMethods(gripper: GripperKind, refresh = 0) {
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  useEffect(() => {
    api.methods(gripper).then(setMethods).catch(() => setMethods([]));
  }, [gripper, refresh]);
  return methods;
}

/**
 * Bir simülasyon oturumu ve 3D animasyonu birlikte yönetir.
 * Adım isteği gelince yeni durum hemen gösterilmez: önce robot animasyonu eski parçalar
 * üzerinde oynar, animasyon bitince (onEventDone) yeni durum uygulanır.
 */
export function useSimulation(gripper: GripperKind, method: string, seed: number, animate = true) {
  const [session, setSession] = useState<SessionState | null>(null);
  const [parts, setParts] = useState<Part[]>([]);
  const [event, setEvent] = useState<GraspEvent | null>(null);
  const [eventKey, setEventKey] = useState(0);
  const [lastEvent, setLastEvent] = useState<GraspEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const pending = useRef<{ state: SessionState; event: GraspEvent } | null>(null);
  const [busyState, setBusyState] = useState(false);

  const setBusy = (b: boolean) => {
    busy.current = b;
    setBusyState(b);
  };

  useEffect(() => {
    let live = true;
    setError(null);
    api.createSession(gripper, method, seed)
      .then((s) => {
        if (!live) return;
        setSession(s);
        setParts(s.parts);
        setEvent(null);
        setLastEvent(null);
        setBusy(false);
      })
      .catch((e) => live && setError(String(e.message ?? e)));
    return () => {
      live = false;
    };
    // seed değişimi reset ile ele alınır; oturum yalnızca tutucu/yöntem değişince yeniden kurulur
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gripper, method]);

  const reset = useCallback(async (newSeed?: number) => {
    if (!session) return;
    setBusy(true);
    try {
      const s = await api.reset(session.id, newSeed);
      setSession(s);
      setParts(s.parts);
      setEvent(null);
      setLastEvent(null);
    } catch (e) {
      setError(String((e as Error).message));
    }
    setBusy(false);
  }, [session]);

  const step = useCallback(async (action?: GraspAction) => {
    if (!session || busy.current || session.done) return null;
    setBusy(true);
    try {
      const res = await api.step(session.id, action);
      // "dur" bir kavrama değildir: robot animasyonu yok, sonuç hemen uygulanır
      if (!animate || res.event.outcome === "stop") {
        setSession(res.state);
        setParts(res.state.parts);
        setLastEvent(res.event);
        setBusy(false);
        return res.event;
      }
      pending.current = { state: res.state, event: res.event };
      setParts(res.parts_before);
      setEvent(res.event);
      setEventKey((k) => k + 1);
      return res.event;
    } catch (e) {
      setError(String((e as Error).message));
      setBusy(false);
      return null;
    }
  }, [session, animate]);

  const onEventDone = useCallback(() => {
    const p = pending.current;
    pending.current = null;
    if (p) {
      setSession(p.state);
      setParts(p.state.parts);
      setLastEvent(p.event);
    }
    setEvent(null);
    setBusy(false);
  }, []);

  return { session, parts, event, eventKey, lastEvent, error, busy: busyState, step, reset, onEventDone };
}

export function useHash(): [string, (h: string) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, "") || "giris";
  const [h, setH] = useState(read);
  useEffect(() => {
    const on = () => {
      setH(read());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [h, (v: string) => (window.location.hash = `/${v}`)];
}
