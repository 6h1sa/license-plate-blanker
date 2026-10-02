import { useEffect, useState } from "react";

/**
 * URL の画像を読み込む。読み込みが終わるまでは、keepPrevious なら前の画像を、そうでなければ null を返す。
 * 消去結果は版が上がるたびに URL が変わるので、更新中も前の結果を表示し続けるのに使う。
 */
export function useLoadedImage(url: string | null, keepPrevious = false): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!keepPrevious) setImage(null);
    if (!url) {
      setImage(null);
      return;
    }
    let alive = true;
    const img = new Image();
    img.onload = () => alive && setImage(img);
    img.src = url;
    return () => {
      alive = false;
    };
  }, [url, keepPrevious]);

  return image;
}
