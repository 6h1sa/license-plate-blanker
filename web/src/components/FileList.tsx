import type { ImageSummary } from "../types";

const STATUS_LABEL: Record<string, string> = { queued: "待機中", detecting: "検出中…", erasing: "消去中…" };

function statusText(f: ImageSummary): { text: string; className: string } {
  if (f.status === "error") return { text: `エラー: ${f.error}`, className: "error" };
  if (f.status === "done") return { text: `完了・ナンバー ${f.n_enabled} 枚`, className: "done" };
  return { text: STATUS_LABEL[f.status] ?? f.status, className: "busy" };
}

interface Props {
  files: ImageSummary[];
  currentId: string | null;
  onSelect(id: string): void;
  onRemove(id: string): void;
}

/** 読み込んだ写真の一覧と、それぞれの処理の進み具合。 */
export function FileList({ files, currentId, onSelect, onRemove }: Props) {
  if (!files.length) {
    return <div className="empty">写真をドラッグ&ドロップするか、「写真を追加」から選んでください。</div>;
  }
  return (
    <>
      {files.map((f) => {
        const st = statusText(f);
        return (
          <div key={f.id} className={`file ${f.id === currentId ? "current" : ""}`} onClick={() => onSelect(f.id)}>
            <div className="name" title={f.name}>{f.name}</div>
            <button
              className="remove"
              title="一覧から外す"
              onClick={(e) => {
                e.stopPropagation();
                onRemove(f.id);
              }}
            >
              ✕
            </button>
            <div className={`status ${st.className}`}>{st.text}</div>
          </div>
        );
      })}
    </>
  );
}
