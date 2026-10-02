/** 画像座標と画面（Canvas の画素）座標の変換、四角形の当たり判定など。React に依存しない純粋な関数。 */

import type { Point, Quad } from "./types";

/** 表示の拡大率 s と、画像の原点が来る画面上の位置 (ox, oy)。 */
export interface View {
  s: number;
  ox: number;
  oy: number;
}

export const toScreen = (v: View, [x, y]: Point): Point => [x * v.s + v.ox, y * v.s + v.oy];
export const toImage = (v: View, [sx, sy]: Point): Point => [(sx - v.ox) / v.s, (sy - v.oy) / v.s];

/** 画像全体が画面に収まる表示。 */
export function fitView(imgW: number, imgH: number, cvW: number, cvH: number, pad = 0.98): View {
  const s = Math.min(cvW / imgW, cvH / imgH) * pad;
  return { s, ox: (cvW - imgW * s) / 2, oy: (cvH - imgH * s) / 2 };
}

/** 画面上の点 (sx, sy) を中心に k 倍する。 */
export function zoomAt(v: View, [sx, sy]: Point, k: number): View {
  return { s: v.s * k, ox: sx - (sx - v.ox) * k, oy: sy - (sy - v.oy) * k };
}

export function quadBounds(q: Quad) {
  const xs = q.map((p) => p[0]);
  const ys = q.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
}

/** ナンバーを画面の中央に、周りが少し見える大きさで表示する。 */
export function zoomToQuad(q: Quad, cvW: number, cvH: number): View {
  const b = quadBounds(q);
  const s = Math.min(cvW / (b.w * 2.2), cvH / (b.h * 3));
  return { s, ox: cvW / 2 - b.cx * s, oy: cvH / 2 - b.cy * s };
}

/** 点が四角形の内側にあるか（交差数による判定）。 */
export function pointInQuad(q: Quad, [x, y]: Point): boolean {
  let inside = false;
  for (let i = 0, j = 3; i < 4; j = i++) {
    const [xi, yi] = q[i];
    const [xj, yj] = q[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** 画面上の点から radius 以内にある角の番号（無ければ -1）。 */
export function cornerAt(q: Quad, v: View, sp: Point, radius: number): number {
  for (let j = 0; j < 4; j++) {
    const [x, y] = toScreen(v, q[j]);
    if (Math.hypot(x - sp[0], y - sp[1]) < radius) return j;
  }
  return -1;
}

/** 2 点で囲んだ範囲 [x1, y1, x2, y2]（左上・右下に並べ替え）。 */
export function boxFrom(a: Point, b: Point): [number, number, number, number] {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}
