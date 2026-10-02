import type { Plate } from "../types";
import { PlateCrop } from "./PlateCrop";

interface Props {
  plates: Plate[];
  selected: number;
  original: HTMLImageElement | null;
  /** 最新の消去結果（更新中は null） */
  result: HTMLImageElement | null;
  onSelect(index: number): void;
  onZoom(index: number): void;
  onToggle(index: number): void;
  onRemove(index: number): void;
}

function sourceLabel(p: Plate): string {
  if (p.source === "manual") return "手動";
  const score = p.score != null ? ` ${p.score.toFixed(2)}` : "";
  return p.source === "auto-edited" ? `自動${score}（修正）` : `自動${score}`;
}

/** ナンバーごとに、処理対象かどうかと、処理前・処理後の拡大を並べる。 */
export function PlatePanel({ plates, selected, original, result, onSelect, onZoom, onToggle, onRemove }: Props) {
  if (!plates.length) {
    return <div className="muted">ナンバーは見つかりませんでした。「囲んで追加」で指定できます。</div>;
  }
  return (
    <>
      {plates.map((p, i) => (
        <div
          key={i}
          className={`plate ${i === selected ? "selected" : ""} ${p.enabled ? "" : "off"}`}
          onClick={() => onSelect(i)}
          onDoubleClick={() => onZoom(i)}
        >
          <div className="plate-head">
            <input type="checkbox" checked={p.enabled} onClick={(e) => e.stopPropagation()} onChange={() => onToggle(i)} />
            <span className="label">#{i + 1} {sourceLabel(p)}</span>
            <button title="削除" onClick={(e) => { e.stopPropagation(); onRemove(i); }}>✕</button>
          </div>
          {!p.enabled && <div className="reason">{p.reason ? `対象外: ${p.reason}` : "対象外"}</div>}
          <div className="crops">
            <PlateCrop image={original} quad={p.quad} placeholder="" />
            <PlateCrop image={result} quad={p.quad} placeholder="更新中…" />
          </div>
        </div>
      ))}
    </>
  );
}
