import { useCallback, useEffect, useRef, useState } from "react";
import { getImage, savePlates } from "../api";
import type { ImageDetail, ImageSummary, Plate } from "../types";

const SAVE_DELAY_MS = 350;

/**
 * 表示中の写真のナンバーを編集する。
 *
 * ナンバーと縁の幅は画面側が持ち、編集のたびに少し待ってからまとめてサーバーへ保存する（サーバーは消去をやり直す）。
 * 処理の進み具合（status・result_version）は一覧の定期取得（summary）から受け取る。
 */
export function useImageEditor(id: string | null, summary: ImageSummary | undefined) {
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [pending, setPending] = useState(false); // 保存待ち・保存中の編集がある
  const latest = useRef<ImageDetail | null>(null);
  const timer = useRef<number | undefined>(undefined);
  latest.current = detail;

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current);
    timer.current = undefined;
    const d = latest.current;
    if (!d) return;
    try {
      const { version } = await savePlates(d.id, d.plates, d.margin);
      setDetail((cur) => (cur && cur.id === d.id ? { ...cur, version } : cur));
    } finally {
      if (timer.current === undefined) setPending(false); // 保存中に新しい編集が無ければ
    }
  }, []);

  const schedule = useCallback(() => {
    setPending(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(flush, SAVE_DELAY_MS);
  }, [flush]);

  // 写真を切り替えたら読み込み直す。検出が終わって大きさが分かったときも、ナンバーを受け取るため読み込み直す
  const detected = summary?.width != null;
  useEffect(() => {
    if (!id) {
      setDetail(null);
      return;
    }
    let alive = true;
    getImage(id).then((d) => alive && setDetail(d), () => alive && setDetail(null));
    return () => {
      alive = false;
      if (timer.current !== undefined) flush(); // 保存待ちの編集は切り替える前に送る
    };
  }, [id, detected, flush]);

  const updatePlates = useCallback(
    (fn: (plates: Plate[]) => Plate[]) => {
      setDetail((d) => (d ? { ...d, plates: fn(d.plates) } : d));
      schedule();
    },
    [schedule],
  );

  const setMargin = useCallback(
    (margin: number) => {
      setDetail((d) => (d ? { ...d, margin } : d));
      schedule();
    },
    [schedule],
  );

  // 処理の進み具合は一覧のほうが新しい。版は保存の応答と一覧の大きいほう
  const current: ImageDetail | null =
    detail && summary && summary.id === detail.id
      ? { ...detail, status: summary.status, error: summary.error, result_version: summary.result_version,
          version: Math.max(detail.version, summary.version) }
      : detail;
  const fresh = current != null && !pending && current.result_version === current.version;

  return { detail: current, updatePlates, setMargin, fresh };
}
