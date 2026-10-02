import { useCallback, useEffect, useState } from "react";
import { listImages } from "../api";
import type { ImageSummary } from "../types";

const POLL_MS = 1000;

/** 読み込んだ写真の一覧。裏の処理の進み具合を知るため、定期的に取り直す。 */
export function useImageList() {
  const [files, setFiles] = useState<ImageSummary[]>([]);

  const refresh = useCallback(async () => {
    try {
      setFiles(await listImages());
    } catch {
      // サーバーが一時的に応答しなくても、次の取得で回復する
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  return { files, refresh };
}
