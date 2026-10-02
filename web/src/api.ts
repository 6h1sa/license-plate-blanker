/** app.py の API を呼ぶ関数。失敗したらサーバーのメッセージ付きで例外を投げる。 */

import type { ImageDetail, ImageSummary, Plate, Quad, SaveFormat } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const listImages = () => request<ImageSummary[]>("/api/images");

export const getImage = (id: string) => request<ImageDetail>(`/api/image/${id}`);

export const uploadImage = (file: File) =>
  request<{ id: string }>("/api/upload", {
    method: "POST",
    headers: { "X-Filename": encodeURIComponent(file.name) },
    body: file,
  });

export const deleteImage = (id: string) => request<{ ok: boolean }>(`/api/image/${id}`, { method: "DELETE" });

/** 編集を保存する。サーバーは版を上げて消去をやり直し、新しい版を返す。 */
export const savePlates = (id: string, plates: Plate[], margin: number) =>
  request<{ version: number }>(`/api/image/${id}/plates`, {
    method: "POST",
    body: JSON.stringify({ plates, margin }),
  });

/** 手で囲んだ範囲 [x1, y1, x2, y2] から四隅を推定する（推定できなければ囲んだ四角形が返る）。 */
export const refineBox = (id: string, box: [number, number, number, number]) =>
  request<{ quad: Quad; ok: boolean; reason: string }>(`/api/image/${id}/refine`, {
    method: "POST",
    body: JSON.stringify({ box }),
  });

/** 表示用の画像の URL。結果は版をクエリに入れて、古い画像がキャッシュされないようにする。 */
export const viewUrl = (id: string) => `/img/${id}/view`;
export const resultUrl = (id: string, version: number) => `/img/${id}/result?v=${version}`;

/** 元の解像度で消去したファイル（target が "all" なら ZIP）を取得する。 */
export async function download(target: string, format: SaveFormat): Promise<{ blob: Blob; name: string }> {
  const res = await fetch(`/api/download/${target}?format=${format}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? res.statusText);
  }
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename\*=UTF-8''([^;]+)/);
  return { blob: await res.blob(), name: match ? decodeURIComponent(match[1]) : "download" };
}
