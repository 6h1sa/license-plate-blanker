import { useCallback, useEffect, useState } from "react";
import { eventsUrl, listImages } from "../api";
import type { ImageSummary } from "../types";

/**
 * 読み込んだ写真の一覧。
 *
 * 定期的には問い合わせず、サーバーが状態の変化を知らせてきたとき（/api/events）だけ取り直す。
 * 接続が切れても EventSource が自動でつなぎ直し、つながった直後の通知で最新になる。
 */
export function useImageList() {
  const [files, setFiles] = useState<ImageSummary[]>([]);

  const refresh = useCallback(async () => {
    try {
      setFiles(await listImages());
    } catch {
      // サーバーが一時的に応答しなくても、次の通知で回復する
    }
  }, []);

  useEffect(() => {
    const events = new EventSource(eventsUrl());
    events.onmessage = () => refresh();
    return () => events.close();
  }, [refresh]);

  return { files, refresh };
}
