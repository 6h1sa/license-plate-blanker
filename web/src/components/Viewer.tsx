import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import { boxFrom, cornerAt, fitView, pointInQuad, toImage, toScreen, zoomAt, zoomToQuad, type View } from "../geometry";
import type { Plate, Point, Quad, ToolKind, ViewMode } from "../types";

const COLORS = { enabled: "#4caf50", disabled: "#bbbbbb", selected: "#4fc3f7", tool: "#ff4081", background: "#101114" };
const HANDLE_RADIUS = 6;
const HIT_RADIUS = 9;

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
  /** 除外された候補をクリックしたとき */
  onEnable(index: number): void;
  /** 四隅のドラッグが終わったとき */
  onMoveCorner(index: number, quad: Quad): void;
  /** 範囲で囲み終えた（box）、または四隅を置き終えた（quad）とき */
  onToolDone(kind: ToolKind, points: Point[]): void;
}

type Drag =
  | { type: "pan"; start: Point; origin: View }
  | { type: "corner"; corner: number; quad: Quad }
  | { type: "box"; start: Point; current: Point };

/** 写真を拡大・移動して表示し、ナンバーの枠の選択・四隅の修正・手での追加を受け付ける Canvas。 */
export const Viewer = forwardRef<ViewerHandle, Props>(function Viewer(props, ref) {
  const { original, result, mode, plates, selected, tool } = props;
  const canvas = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View>({ s: 1, ox: 0, oy: 0 });
  const [drag, setDrag] = useState<Drag | null>(null);
  const [quadPoints, setQuadPoints] = useState<Point[]>([]);
  const [hoverCorner, setHoverCorner] = useState(-1);
  const dpr = window.devicePixelRatio || 1;

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
      const r = c.getBoundingClientRect();
      c.width = r.width * dpr;
      c.height = r.height * dpr;
      setView((v) => ({ ...v })); // 描き直す
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
      const quad = drag?.type === "corner" && isSel ? drag.quad : p.quad;
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
          g.arc(x, y, HANDLE_RADIUS * dpr, 0, Math.PI * 2);
          g.fillStyle = j === hoverCorner ? "#ffffff" : COLORS.selected + "aa";
          g.fill();
          g.strokeStyle = "#000";
          g.lineWidth = dpr;
          g.stroke();
        });
      }
    });

    if (drag?.type === "box") {
      const [a, b] = [toScreen(view, drag.start), toScreen(view, drag.current)];
      g.strokeStyle = COLORS.tool;
      g.lineWidth = 1.5 * dpr;
      g.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    }
    g.fillStyle = COLORS.tool;
    for (const pt of quadPoints) {
      const [x, y] = toScreen(view, pt);
      g.beginPath();
      g.arc(x, y, 4 * dpr, 0, Math.PI * 2);
      g.fill();
    }
  }, [original, result, mode, plates, selected, view, drag, quadPoints, hoverCorner, dpr]);

  // --- マウス
  const screenPoint = (e: { clientX: number; clientY: number }): Point => {
    const r = canvas.current!.getBoundingClientRect();
    return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr];
  };
  const plateAt = (pt: Point) => plates.findIndex((p) => pointInQuad(p.quad, pt));

  const onMouseDown = (e: React.MouseEvent) => {
    if (!original || e.button !== 0) return;
    const sp = screenPoint(e);
    const pt = toImage(view, sp);
    if (tool === "box") return setDrag({ type: "box", start: pt, current: pt });
    if (tool === "quad") {
      const pts = [...quadPoints, pt];
      if (pts.length === 4) {
        setQuadPoints([]);
        props.onToolDone("quad", pts);
      } else setQuadPoints(pts);
      return;
    }
    if (selected >= 0 && plates[selected]) {
      const corner = cornerAt(plates[selected].quad, view, sp, HIT_RADIUS * dpr);
      if (corner >= 0) return setDrag({ type: "corner", corner, quad: plates[selected].quad });
    }
    const hit = plateAt(pt);
    if (hit >= 0 && !plates[hit].enabled && hit !== selected) props.onEnable(hit);
    props.onSelect(hit);
    setDrag({ type: "pan", start: sp, origin: view });
  };

  // ドラッグ中はキャンバスの外に出ても追うため window で受ける
  useEffect(() => {
    if (!drag) return;
    const move = (e: MouseEvent) => {
      const sp = screenPoint(e);
      if (drag.type === "pan") {
        setView({ ...drag.origin, ox: drag.origin.ox + sp[0] - drag.start[0], oy: drag.origin.oy + sp[1] - drag.start[1] });
      } else if (drag.type === "corner") {
        const quad = drag.quad.map((p, j) => (j === drag.corner ? toImage(view, sp) : p)) as Quad;
        setDrag({ ...drag, quad });
      } else {
        setDrag({ ...drag, current: toImage(view, sp) });
      }
    };
    const up = () => {
      setDrag(null);
      if (drag.type === "corner" && drag.quad !== plates[selected]?.quad) props.onMoveCorner(selected, drag.quad);
      if (drag.type === "box") {
        const [x1, y1, x2, y2] = boxFrom(drag.start, drag.current);
        if ((x2 - x1) * view.s >= 8 && (y2 - y1) * view.s >= 6) props.onToolDone("box", [[x1, y1], [x2, y2]]);
      }
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  });

  const onHover = (e: React.MouseEvent) => {
    if (drag || selected < 0 || !plates[selected]) return;
    const c = cornerAt(plates[selected].quad, view, screenPoint(e), HIT_RADIUS * dpr);
    if (c !== hoverCorner) setHoverCorner(c);
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    const hit = plateAt(toImage(view, screenPoint(e)));
    if (hit >= 0) setView(zoomToQuad(plates[hit].quad, ...size()));
  };

  // ページがスクロールしないよう passive: false で受ける
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
  const hint = tool === "box" ? "ナンバーをドラッグで囲んでください（Esc で取り消し）"
    : tool === "quad" ? `ナンバーの四隅をクリック（${quadPoints.length}/4、Esc で取り消し）` : null;

  return (
    <>
      <canvas ref={canvas} className="viewer" style={{ cursor }} onMouseDown={onMouseDown} onMouseMove={onHover} onDoubleClick={onDoubleClick} />
      {original && (
        <div className="badge">
          <span className="mode">{mode === "after" ? "処理後" : "処理前"}</span>
          <span>拡大 {((view.s / dpr) * 100).toFixed(0)}%</span>
          {props.status && <span>{props.status}</span>}
        </div>
      )}
      {hint && <div className="hint">{hint}</div>}
    </>
  );
});
