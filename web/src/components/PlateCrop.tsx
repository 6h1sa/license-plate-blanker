import { useLayoutEffect, useRef } from "react";
import { quadBounds } from "../geometry";
import type { Quad } from "../types";

interface Props {
  image: HTMLImageElement | null;
  quad: Quad;
  /** 画像がまだ無いときの文言 */
  placeholder: string;
}

/** 画像からナンバーの周りを切り出して表示する。 */
export function PlateCrop({ image, quad, placeholder }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const c = canvas.current!;
    const dpr = window.devicePixelRatio || 1;
    const r = c.getBoundingClientRect();
    c.width = r.width * dpr;
    c.height = r.height * dpr;
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, c.width, c.height);
    if (!image) {
      g.fillStyle = "#666";
      g.font = `${11 * dpr}px sans-serif`;
      g.fillText(placeholder, 6 * dpr, 16 * dpr);
      return;
    }
    const b = quadBounds(quad);
    const x0 = b.x0 - b.w * 0.15, y0 = b.y0 - b.h * 0.25, w = b.w * 1.3, h = b.h * 1.5;
    const s = Math.min(c.width / w, c.height / h);
    g.drawImage(image, x0, y0, w, h, (c.width - w * s) / 2, (c.height - h * s) / 2, w * s, h * s);
  }, [image, quad, placeholder]);

  return <canvas ref={canvas} className="crop" />;
}
