import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import {
  boxFrom, cornerAt, distance, fitView, midpoint, pinchView, pointInQuad, toImage, toScreen, zoomAt, zoomToQuad, type View,
} from "../geometry";
import { COARSE_POINTER, useMediaQuery } from "../hooks/useMediaQuery";
import type { Plate, Point, Quad, ToolKind, ViewMode } from "../types";

const COLORS = { enabled: "#4caf50", disabled: "#bbbbbb", selected: "#4fc3f7", tool: "#ff4081", background: "#101114" };
/** 角の丸の半径と、つかめる範囲（CSS ピクセル）。指で操作する端末では大きくする */
const HANDLE = { mouse: { radius: 6, hit: 9 }, touch: { radius: 10, hit: 24 } };
/** これより動いたらタップではなくドラッグとみなす（CSS ピクセル） */
const TAP_SLOP = { mouse: 4, touch: 10 };
/** ダブルタップとみなす間隔（ミリ秒）と距離（CSS ピクセル） */
const DOUBLE_TAP = { ms: 300, px: 30 };
/** 指で角を動かすときに出す拡大表示（ルーペ）。半径と、指からの距離（CSS ピクセル）、今の表示に対する倍率 */
const LOUPE = { radius: 60, gap: 40, zoom: 3 };

export interface ViewerHandle {
  fit(): void;
  zoomTo(quad: Quad): void;
}

interface Props {
  /** 処理前の画像（ナンバーの座標の基準） */
  original: HTMLImageElement | null;
  /** 処理後の画像（まだ無ければ null） */
  result: HTMLImageElement | null;
  mode: ViewMode;
  plates: Plate[];
  selected: number;
  tool: ToolKind | null;
  status: string;
  onSelect(index: number): void;
  /** 除外された候補をタップしたとき */
  onEnable(index: number): void;
  /** 四隅のドラッグが終わったとき */
  onMoveCorner(index: number, quad: Quad): void;
  /** 範囲で囲み終えた（box）、または四隅を置き終えた（quad）とき */
  onToolDone(kind: ToolKind, points: Point[]): void;
  onCancelTool(): void;
}

/** 指（またはマウス）の操作の途中の状態。座標は Canvas の画素。 */
type Gesture =
  /** 1 本指を置いた直後。動けば移動、動かずに離せばタップ */
  | { kind: "pending"; id: number; start: Point; origin: View }
  | { kind: "pan"; id: number; start: Point; origin: View }
  /** 角のドラッグ。角は指を置いた位置ではなく、指を動かした分だけ動かす（つかんだ瞬間に飛ばない） */
  | { kind: "corner"; id: number; corner: number; quad: Quad; start: Point; from: Point; touch: boolean }
  | { kind: "box"; id: number; start: Point; current: Point; touch: boolean }
  /** 指で「4隅で追加」の点を置く途中。指を動かして位置を合わせ、離すと確定する */
  | { kind: "place"; id: number; point: Point; start: Point; from: Point }
  | { kind: "pinch"; startMid: Point; startDist: number; origin: View };

/** 写真を拡大・移動して表示し、ナンバーの枠の選択・四隅の修正・手での追加を受け付ける Canvas。マウスと指の両方で操作できる。 */
export const Viewer = forwardRef<ViewerHandle, Props>(function Viewer(props, ref) {
  const { original, result, mode, plates, selected, tool } = props;
  const canvas = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View>({ s: 1, ox: 0, oy: 0 });
  const [gesture, setGesture] = useState<Gesture | null>(null);
  const [quadPoints, setQuadPoints] = useState<Point[]>([]);
  const [hoverCorner, setHoverCorner] = useState(-1);
  const coarse = useMediaQuery(COARSE_POINTER);
  const dpr = window.devicePixelRatio || 1;
  const handle = coarse ? HANDLE.touch : HANDLE.mouse;

  // イベントの処理は最新の値を見る必要があるので、ref にも持つ
  const pointers = useRef(new Map<number, Point>());
  const lastTap = useRef<{ time: number; at: Point } | null>(null);
  const latest = useRef({ view, gesture, props, quadPoints });
  latest.current = { view, gesture, props, quadPoints };

  const size = () => {
    const c = canvas.current!;
    return [c.width, c.height] as const;
  };
  const fit = useCallback(() => {
    if (original) setView(fitView(original.width, original.height, ...size()));
  }, [original]);
  useImperativeHandle(ref, () => ({ fit, zoomTo: (q) => setView(zoomToQuad(q, ...size())) }), [fit]);

  // Canvas の画素数を表示の大きさに合わせる
  useLayoutEffect(() => {
    const c = canvas.current!;
    const resize = () => {
      // 大きさが変わっても（パネルの開閉・画面の回転）、それまで中央に映っていた場所を中央に保つ
      const [ow, oh] = [c.width, c.height];
      const r = c.getBoundingClientRect();
      c.width = r.width * dpr;
      c.height = r.height * dpr;
      setView((v) => ({ ...v, ox: v.ox + (c.width - ow) / 2, oy: v.oy + (c.height - oh) / 2 }));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(c);
    return () => ro.disconnect();
  }, [dpr]);

  // 新しい写真を表示したら全体に合わせる
  useEffect(fit, [fit]);
  // 追加の操作を始めた・やめたら、置きかけの点を消す
  useEffect(() => setQuadPoints([]), [tool]);

  // --- 描画
  useEffect(() => {
    const c = canvas.current!;
    const g = c.getContext("2d")!;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = COLORS.background;
    g.fillRect(0, 0, c.width, c.height);
    if (!original) return;

    const img = mode === "after" && result ? result : original;
    g.imageSmoothingEnabled = view.s < 2;
    g.drawImage(img, view.ox, view.oy, original.width * view.s, original.height * view.s);

    const polygon = (q: Point[], color: string, width: number, dash: number[] = []) => {
      g.beginPath();
      q.forEach((p, i) => {
        const [x, y] = toScreen(view, p);
        if (i) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      g.closePath();
      g.setLineDash(dash);
      g.strokeStyle = color;
      g.lineWidth = width * dpr;
      g.stroke();
      g.setLineDash([]);
    };

    plates.forEach((p, i) => {
      const isSel = i === selected;
      if (mode === "after" && !isSel) return; // 処理後は選んだものだけ枠を出す
      const quad = gesture?.kind === "corner" && isSel ? gesture.quad : p.quad;
      const color = isSel ? COLORS.selected : p.enabled ? COLORS.enabled : COLORS.disabled;
      polygon(quad, color, isSel ? 2.5 : 1.5, p.enabled ? [] : [6, 4]);
      const [lx, ly] = toScreen(view, quad[0]);
      g.font = `${12 * dpr}px sans-serif`;
      g.fillStyle = color;
      g.fillText(`#${i + 1}`, lx, ly - 4 * dpr);
      if (isSel) {
        quad.forEach((pt, j) => {
          const [x, y] = toScreen(view, pt);
          g.beginPath();
          g.arc(x, y, handle.radius * dpr, 0, Math.PI * 2);
          const active = j === hoverCorner || (gesture?.kind === "corner" && gesture.corner === j);
          g.fillStyle = active ? "#ffffff" : COLORS.selected + "aa";
          g.fill();
          g.strokeStyle = "#000";
          g.lineWidth = dpr;
          g.stroke();
        });
      }
    });

    if (gesture?.kind === "box") {
      const [a, b] = [toScreen(view, gesture.start), toScreen(view, gesture.current)];
      g.strokeStyle = COLORS.tool;
      g.lineWidth = 1.5 * dpr;
      g.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    }
    g.fillStyle = COLORS.tool;
    for (const pt of gesture?.kind === "place" ? [...quadPoints, gesture.point] : quadPoints) {
      const [x, y] = toScreen(view, pt);
      g.beginPath();
      g.arc(x, y, 4 * dpr, 0, Math.PI * 2);
      g.fill();
    }

    // 指の下は見えないので、指から離れた位置に、いじっている点の周りを拡大して出す
    const focus: Point | null = gesture?.kind === "corner" && gesture.touch ? gesture.quad[gesture.corner]
      : gesture?.kind === "box" && gesture.touch ? gesture.current
      : gesture?.kind === "place" ? gesture.point : null;
    if (focus) {
      const finger = toScreen(view, focus);
      const R = LOUPE.radius * dpr;
      const gap = LOUPE.gap * dpr;
      // 基本は指の上。上に入らなければ、画面の中央寄りの横に出す
      let cx = finger[0];
      let cy = finger[1] - R - gap;
      if (cy - R < 0) {
        cx = finger[0] > c.width / 2 ? finger[0] - R - gap : finger[0] + R + gap;
        cy = Math.min(Math.max(finger[1], R), c.height - R);
      }
      cx = Math.min(Math.max(cx, R), c.width - R);
      const ls = Math.min(view.s * LOUPE.zoom, 8 * dpr); // 画像の 1 画素を最大 8 CSS ピクセルまで拡大
      const L = (pt: Point): [number, number] => [cx + (pt[0] - focus[0]) * ls, cy + (pt[1] - focus[1]) * ls];
      g.save();
      g.beginPath();
      g.arc(cx, cy, R, 0, Math.PI * 2);
      g.clip();
      g.fillStyle = COLORS.background;
      g.fillRect(cx - R, cy - R, 2 * R, 2 * R);
      g.imageSmoothingEnabled = ls < 2 * dpr;
      g.drawImage(img, focus[0] - R / ls, focus[1] - R / ls, (2 * R) / ls, (2 * R) / ls, cx - R, cy - R, 2 * R, 2 * R);
      const outline: Point[] | null = gesture?.kind === "corner" ? gesture.quad
        : gesture?.kind === "box" ? [gesture.start, [gesture.current[0], gesture.start[1]], gesture.current, [gesture.start[0], gesture.current[1]]]
        : null;
      if (outline) {
        g.beginPath();
        outline.forEach((pt, i) => {
          const [x, y] = L(pt);
          if (i) g.lineTo(x, y);
          else g.moveTo(x, y);
        });
        g.closePath();
        g.strokeStyle = gesture?.kind === "box" ? COLORS.tool : COLORS.selected;
        g.lineWidth = 1.5 * dpr;
        g.stroke();
      }
      // 中心の十字 = いじっている点の位置
      const a = 3 * dpr;
      const b = 10 * dpr;
      g.strokeStyle = "#ffffff";
      g.lineWidth = dpr;
      g.beginPath();
      g.moveTo(cx - b, cy);
      g.lineTo(cx - a, cy);
      g.moveTo(cx + a, cy);
      g.lineTo(cx + b, cy);
      g.moveTo(cx, cy - b);
      g.lineTo(cx, cy - a);
      g.moveTo(cx, cy + a);
      g.lineTo(cx, cy + b);
      g.stroke();
      g.restore();
      g.beginPath();
      g.arc(cx, cy, R, 0, Math.PI * 2);
      g.strokeStyle = COLORS.selected;
      g.lineWidth = 2 * dpr;
      g.stroke();
    }
  }, [original, result, mode, plates, selected, view, gesture, quadPoints, hoverCorner, dpr, handle.radius]);

  // --- 指とマウスの操作（Pointer Events）
  const screenPoint = (e: { clientX: number; clientY: number }): Point => {
    const r = canvas.current!.getBoundingClientRect();
    return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr];
  };
  const isTouch = (e: React.PointerEvent) => e.pointerType !== "mouse";
  /** 指を start から sp へ動かした分だけ、画像上の点 from を動かす */
  const moveBy = (from: Point, start: Point, sp: Point, scale: number): Point =>
    [from[0] + (sp[0] - start[0]) / scale, from[1] + (sp[1] - start[1]) / scale];

  const startPinch = () => {
    const [a, b] = [...pointers.current.values()];
    setGesture({ kind: "pinch", startMid: midpoint(a, b), startDist: distance(a, b), origin: latest.current.view });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!original || (e.pointerType === "mouse" && e.button !== 0)) return;
    canvas.current!.setPointerCapture(e.pointerId);
    const sp = screenPoint(e);
    pointers.current.set(e.pointerId, sp);

    // 2 本目の指: 途中の 1 本指の操作は取りやめて、拡大縮小に切り替える
    if (pointers.current.size === 2) return startPinch();
    if (pointers.current.size > 2) return;

    const v = latest.current.view;
    if (tool === "box") {
      return setGesture({ kind: "box", id: e.pointerId, start: toImage(v, sp), current: toImage(v, sp), touch: isTouch(e) });
    }
    if (tool === "quad" && isTouch(e)) {
      return setGesture({ kind: "place", id: e.pointerId, point: toImage(v, sp), start: sp, from: toImage(v, sp) });
    }
    if (tool !== "quad" && selected >= 0 && plates[selected]) {
      const q = plates[selected].quad;
      const corner = cornerAt(q, v, sp, (isTouch(e) ? HANDLE.touch : HANDLE.mouse).hit * dpr);
      if (corner >= 0) {
        return setGesture({ kind: "corner", id: e.pointerId, corner, quad: q, start: sp, from: q[corner], touch: isTouch(e) });
      }
    }
    setGesture({ kind: "pending", id: e.pointerId, start: sp, origin: v });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const sp = screenPoint(e);
    const { gesture: gs, view: v } = latest.current;
    if (!pointers.current.has(e.pointerId)) {
      // ボタンを押していないマウスの動き: 角の上なら強調する
      if (e.pointerType === "mouse" && selected >= 0 && plates[selected]) {
        const c = cornerAt(plates[selected].quad, v, sp, HANDLE.mouse.hit * dpr);
        if (c !== hoverCorner) setHoverCorner(c);
      }
      return;
    }
    pointers.current.set(e.pointerId, sp);
    if (!gs) return;
    if (gs.kind === "pinch") {
      if (pointers.current.size < 2) return;
      const [a, b] = [...pointers.current.values()];
      setView(pinchView(gs.origin, gs.startMid, gs.startDist, midpoint(a, b), distance(a, b)));
      return;
    }
    if (gs.id !== e.pointerId) return;
    if (gs.kind === "pending") {
      const slop = (isTouch(e) ? TAP_SLOP.touch : TAP_SLOP.mouse) * dpr;
      if (distance(sp, gs.start) > slop) setGesture({ kind: "pan", id: gs.id, start: gs.start, origin: gs.origin });
    } else if (gs.kind === "pan") {
      setView({ ...gs.origin, ox: gs.origin.ox + sp[0] - gs.start[0], oy: gs.origin.oy + sp[1] - gs.start[1] });
    } else if (gs.kind === "corner") {
      const moved = moveBy(gs.from, gs.start, sp, v.s);
      setGesture({ ...gs, quad: gs.quad.map((p, j) => (j === gs.corner ? moved : p)) as Quad });
    } else if (gs.kind === "place") {
      setGesture({ ...gs, point: moveBy(gs.from, gs.start, sp, v.s) });
    } else if (gs.kind === "box") {
      setGesture({ ...gs, current: toImage(v, sp) });
    }
  };

  const addQuadPoint = (pt: Point) => {
    const { props: p, quadPoints: qp } = latest.current;
    const pts = [...qp, pt];
    if (pts.length === 4) {
      setQuadPoints([]);
      p.onToolDone("quad", pts);
    } else setQuadPoints(pts);
  };

  const onTap = (sp: Point) => {
    const { view: v, props: p } = latest.current;
    const pt = toImage(v, sp);
    if (p.tool === "quad") return addQuadPoint(pt);
    const hit = p.plates.findIndex((pl) => pointInQuad(pl.quad, pt));
    // ダブルタップ（ダブルクリック）: ナンバーなら拡大する
    const now = performance.now();
    const prev = lastTap.current;
    lastTap.current = { time: now, at: sp };
    if (prev && now - prev.time < DOUBLE_TAP.ms && distance(prev.at, sp) < DOUBLE_TAP.px * dpr && hit >= 0) {
      lastTap.current = null;
      setView(zoomToQuad(p.plates[hit].quad, ...size()));
      return;
    }
    if (hit >= 0 && !p.plates[hit].enabled && hit !== p.selected) p.onEnable(hit);
    p.onSelect(hit);
  };

  const onPointerEnd = (e: React.PointerEvent, cancelled: boolean) => {
    if (!pointers.current.delete(e.pointerId)) return;
    const { gesture: gs, view: v, props: p } = latest.current;
    if (!gs) return;
    if (gs.kind === "pinch") {
      // 1 本だけ離したら、残った指で移動を続ける
      const rest = [...pointers.current.entries()];
      setGesture(rest.length === 1 ? { kind: "pan", id: rest[0][0], start: rest[0][1], origin: v } : null);
      return;
    }
    if (gs.id !== e.pointerId) return;
    setGesture(null);
    if (cancelled) return;
    if (gs.kind === "pending") onTap(gs.start);
    else if (gs.kind === "place") addQuadPoint(gs.point);
    else if (gs.kind === "corner" && gs.quad !== p.plates[p.selected]?.quad) p.onMoveCorner(p.selected, gs.quad);
    else if (gs.kind === "box") {
      const [x1, y1, x2, y2] = boxFrom(gs.start, gs.current);
      if ((x2 - x1) * v.s >= 8 && (y2 - y1) * v.s >= 6) p.onToolDone("box", [[x1, y1], [x2, y2]]);
    }
  };

  // ホイールで拡大縮小。ページがスクロールしないよう passive: false で受ける
  useEffect(() => {
    const c = canvas.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const sp = screenPoint(e);
      setView((v) => zoomAt(v, sp, Math.exp(-e.deltaY * 0.0015)));
    };
    c.addEventListener("wheel", wheel, { passive: false });
    return () => c.removeEventListener("wheel", wheel);
  }, [dpr]);

  const cursor = tool ? "crosshair" : hoverCorner >= 0 ? "move" : "default";
  const verb = coarse ? "タップ" : "クリック";
  const hint = tool === "box" ? "ナンバー全体が入るよう、少し外側までドラッグで囲んでください"
    : tool === "quad" && coarse ? `板の外周の角に指を置き、拡大表示で合わせて離す（${quadPoints.length}/4）`
    : tool === "quad" ? `板の外周の角を${verb}（${quadPoints.length}/4）` : null;

  return (
    <>
      <canvas
        ref={canvas}
        className="viewer"
        style={{ cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(e) => onPointerEnd(e, false)}
        onPointerCancel={(e) => onPointerEnd(e, true)}
        onPointerLeave={() => hoverCorner >= 0 && setHoverCorner(-1)}
        onContextMenu={(e) => e.preventDefault()}
      />
      {original && (
        <div className="badge">
          <span className="mode">{mode === "after" ? "処理後" : "処理前"}</span>
          <span className="zoom">拡大 {((view.s / dpr) * 100).toFixed(0)}%</span>
          {props.status && <span>{props.status}</span>}
        </div>
      )}
      {hint && (
        <div className="hint">
          {hint}
          <button onClick={props.onCancelTool}>取り消し{!coarse && <kbd>Esc</kbd>}</button>
        </div>
      )}
    </>
  );
});
