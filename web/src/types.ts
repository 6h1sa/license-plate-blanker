/** サーバー（app.py）とやり取りするデータの型。 */

/** 画像上の点 [x, y]（画素）。 */
export type Point = [number, number];

/** ナンバーの四隅。左上・右上・右下・左下の順。 */
export type Quad = [Point, Point, Point, Point];

export type Status = "queued" | "detecting" | "erasing" | "done" | "error";

export type PlateSource = "auto" | "auto-edited" | "manual";

export interface Plate {
  quad: Quad;
  /** 消去の対象か（チェックボックス） */
  enabled: boolean;
  /** 自動判定でナンバーとみなしたか */
  ok: boolean;
  /** 検出スコア（手動で追加したものは null） */
  score: number | null;
  /** 自動判定で除外した理由 */
  reason: string;
  source: PlateSource;
}

export interface ImageSummary {
  id: string;
  name: string;
  status: Status;
  error: string;
  /** 検出が終わるまでは null */
  width: number | null;
  height: number | null;
  /** 編集するたびに上がる版 */
  version: number;
  /** 表示できる消去結果の版（まだ無ければ -1） */
  result_version: number;
  /** 縁として残す幅（プレート高さに対する %） */
  margin: number;
  n_enabled: number;
}

export interface ImageDetail extends ImageSummary {
  plates: Plate[];
}

export type SaveFormat = "jpeg" | "webp";

/** 表示の切り替え */
export type ViewMode = "before" | "after";

/** ナンバーを手で追加する操作 */
export type ToolKind = "box" | "quad";
