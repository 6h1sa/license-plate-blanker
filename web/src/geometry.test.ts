import { describe, expect, it } from "vitest";
import { boxFrom, cornerAt, fitView, pointInQuad, toImage, toScreen, zoomAt, zoomToQuad } from "./geometry";
import type { Quad } from "./types";

const quad: Quad = [[100, 100], [300, 110], [295, 210], [105, 200]];

describe("座標の変換", () => {
  it("画面と画像の座標は行き来できる", () => {
    const v = { s: 0.5, ox: 20, oy: -10 };
    expect(toImage(v, toScreen(v, [123, 456]))).toEqual([123, 456]);
  });

  it("全体表示は画像が画面に収まり、中央に来る", () => {
    const v = fitView(4000, 2000, 1000, 800, 1);
    expect(v.s).toBe(0.25);
    expect(toScreen(v, [0, 0])).toEqual([0, 150]);
    expect(toScreen(v, [4000, 2000])).toEqual([1000, 650]);
  });

  it("拡大しても、拡大の中心にある画像の点は動かない", () => {
    const v = { s: 1, ox: 0, oy: 0 };
    const z = zoomAt(v, [200, 100], 2);
    expect(toImage(z, [200, 100])).toEqual([200, 100]);
    expect(z.s).toBe(2);
  });

  it("ナンバーへの拡大は、その中心を画面の中央に置く", () => {
    const v = zoomToQuad(quad, 1000, 800);
    const [x, y] = toScreen(v, [200, 155]);
    expect(x).toBeCloseTo(500);
    expect(y).toBeCloseTo(400);
  });
});

describe("当たり判定", () => {
  it("四角形の内側と外側を区別する", () => {
    expect(pointInQuad(quad, [200, 150])).toBe(true);
    expect(pointInQuad(quad, [90, 150])).toBe(false);
    expect(pointInQuad(quad, [200, 220])).toBe(false);
  });

  it("近くの角だけを拾う", () => {
    const v = { s: 1, ox: 0, oy: 0 };
    expect(cornerAt(quad, v, [298, 112], 9)).toBe(1);
    expect(cornerAt(quad, v, [200, 150], 9)).toBe(-1);
  });

  it("囲んだ範囲は向きに関係なく左上・右下になる", () => {
    expect(boxFrom([50, 80], [10, 20])).toEqual([10, 20, 50, 80]);
  });
});
