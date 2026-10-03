import { useEffect, useState } from "react";

/** メディアクエリに今当てはまるか。画面の幅や向きが変わったら更新する。 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatches(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, [query]);

  return matches;
}

/** 狭い画面（スマホなど。横向きのスマホは高さで判定する）。写真を大きく出し、ほかの操作はタブにまとめる */
export const NARROW = "(max-width: 800px), (max-height: 500px)";
/** マウスが無い（指で操作する）端末。キーボードの案内を出さず、つかむ部分を大きくする */
export const COARSE_POINTER = "(pointer: coarse)";
