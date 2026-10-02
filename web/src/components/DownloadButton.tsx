import { forwardRef, useState } from "react";
import { download } from "../api";
import type { SaveFormat } from "../types";

interface Props {
  /** 写真の id、またはすべてなら "all" */
  target: string | null;
  format: SaveFormat;
  className?: string;
  children: React.ReactNode;
}

/** 元の解像度で消去し直してから保存するので時間がかかる。終わるまで「準備中…」と出して押せなくする。 */
export const DownloadButton = forwardRef<HTMLButtonElement, Props>(function DownloadButton({ target, format, className, children }, ref) {
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!target || busy) return;
    setBusy(true);
    try {
      const { blob, name } = await download(target, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (err) {
      alert(`ダウンロードできませんでした: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button ref={ref} className={className} disabled={!target || busy} onClick={run}>
      {busy ? "準備中…" : children}
    </button>
  );
});
