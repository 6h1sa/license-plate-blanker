import { useCallback, useEffect, useRef, useState } from "react";
import { deleteImage, refineBox, resultUrl, uploadImage, viewUrl } from "./api";
import { DownloadButton } from "./components/DownloadButton";
import { FileList } from "./components/FileList";
import { PlatePanel } from "./components/PlatePanel";
import { Viewer, type ViewerHandle } from "./components/Viewer";
import { useImageEditor } from "./hooks/useImageEditor";
import { useImageList } from "./hooks/useImageList";
import { useKeyboard } from "./hooks/useKeyboard";
import { useLoadedImage } from "./hooks/useLoadedImage";
import { COARSE_POINTER, NARROW, useMediaQuery } from "./hooks/useMediaQuery";
import type { Plate, Point, Quad, SaveFormat, ToolKind, ViewMode } from "./types";

const ACCEPT = ".jpg,.jpeg,.png,.webp,.tif,.tiff";
const STATUS_LABEL: Record<string, string> = { queued: "待機中", detecting: "検出中…", erasing: "消去中…", error: "エラー" };

/** 狭い画面で下に出すパネル。null なら閉じて写真を広く見せる */
type Panel = "files" | "plates" | null;

export function App() {
  const { files, refresh } = useImageList();
  const [currentId, setCurrentId] = useState<string | null>(null);
  const summary = files.find((f) => f.id === currentId);
  const { detail, updatePlates, setMargin, fresh } = useImageEditor(currentId, summary);

  const [mode, setMode] = useState<ViewMode>("before");
  const [selected, setSelected] = useState(-1);
  const [tool, setTool] = useState<ToolKind | null>(null);
  const [format, setFormat] = useState<SaveFormat>("jpeg");
  const [dragOver, setDragOver] = useState(false);
  const viewer = useRef<ViewerHandle>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const downloadOne = useRef<HTMLButtonElement>(null);
  const narrow = useMediaQuery(NARROW);
  const coarse = useMediaQuery(COARSE_POINTER);
  const [panel, setPanel] = useState<Panel>(null); // 最初は写真を広く見せる

  const ready = detail?.width != null;
  const original = useLoadedImage(ready && currentId ? viewUrl(currentId) : null);
  const result = useLoadedImage(ready && currentId && detail.result_version >= 0 ? resultUrl(currentId, detail.result_version) : null, true);
  const plates = detail?.plates ?? [];
  // 新しい版の結果が届くまでは、前の版の画像を「最新」として見せない
  const resultIsLatest = fresh && result != null && currentId != null && result.src.endsWith(resultUrl(currentId, detail!.version));

  // 一覧の先頭を自動で開く。開いていた写真が一覧から消えたら閉じる
  useEffect(() => {
    if (currentId && !files.some((f) => f.id === currentId)) setCurrentId(null);
    else if (!currentId && files.length) setCurrentId(files[0].id);
  }, [files, currentId]);
  useEffect(() => {
    setSelected(-1);
    setTool(null);
  }, [currentId]);
  // 狭い画面では、追加の操作中はパネルを閉じて写真を広く使う
  useEffect(() => {
    if (narrow && tool) setPanel(null);
  }, [narrow, tool]);

  // --- ナンバーの編集
  const editPlate = (i: number, patch: Partial<Plate>) => updatePlates((ps) => ps.map((p, j) => (j === i ? { ...p, ...patch } : p)));
  const addPlate = (quad: Quad) => {
    updatePlates((ps) => [...ps, { quad, enabled: true, ok: true, score: null, reason: "", source: "manual" }]);
    setSelected(plates.length);
  };
  const removePlate = (i: number) => {
    updatePlates((ps) => ps.filter((_, j) => j !== i));
    setSelected((s) => (s === i ? -1 : s > i ? s - 1 : s));
  };
  const onToolDone = async (kind: ToolKind, points: Point[]) => {
    setTool(null);
    if (!currentId) return;
    if (kind === "quad") return addPlate(points as Quad);
    const [[x1, y1], [x2, y2]] = points;
    try {
      addPlate((await refineBox(currentId, [x1, y1, x2, y2])).quad);
    } catch (err) {
      alert((err as Error).message);
    }
  };

  // --- 写真の追加・削除
  const upload = useCallback(async (list: File[]) => {
    for (const f of list) {
      try {
        await uploadImage(f);
      } catch (err) {
        alert((err as Error).message);
      }
    }
    refresh();
  }, [refresh]);
  const remove = async (id: string) => {
    await deleteImage(id);
    refresh();
  };
  const step = (d: number) => {
    if (!files.length) return;
    const i = files.findIndex((f) => f.id === currentId);
    setCurrentId(files[(i + d + files.length) % files.length].id);
  };

  useKeyboard({
    " ": () => setMode((m) => (m === "before" ? "after" : "before")),
    f: () => viewer.current?.fit(),
    a: () => ready && setTool("box"),
    q: () => ready && setTool("quad"),
    escape: () => setTool(null),
    n: () => step(1),
    p: () => step(-1),
    d: () => downloadOne.current?.click(),
    x: () => selected >= 0 && editPlate(selected, { enabled: !plates[selected].enabled }),
    delete: () => selected >= 0 && removePlate(selected),
    backspace: () => selected >= 0 && removePlate(selected),
  });

  let status = "";
  if (detail && detail.status !== "done") status = STATUS_LABEL[detail.status] ?? "";
  else if (detail && mode === "after" && !resultIsLatest) status = "更新中…";

  const fileInputEl = (
    <input
      ref={fileInput}
      type="file"
      accept={ACCEPT}
      multiple
      hidden
      onChange={(e) => {
        upload([...(e.target.files ?? [])]);
        e.target.value = "";
      }}
    />
  );

  const viewControls = (
    <div className="row">
      <button className={mode === "before" ? "on" : ""} onClick={() => setMode("before")}>処理前</button>
      <button className={mode === "after" ? "on" : ""} onClick={() => setMode("after")}>処理後</button>
      {!coarse && <kbd>Space</kbd>}
      <button onClick={() => viewer.current?.fit()}>全体{!coarse && <kbd>F</kbd>}</button>
    </div>
  );

  const plateSection = (
    <>
      <div className="row">
        <button disabled={!ready} className={tool === "box" ? "on" : ""} onClick={() => setTool("box")}>囲んで追加{!coarse && <kbd>A</kbd>}</button>
        <button disabled={!ready} className={tool === "quad" ? "on" : ""} onClick={() => setTool("quad")}>4隅で追加{!coarse && <kbd>Q</kbd>}</button>
      </div>
      <div className="plates">
        {detail && !ready && <div className="muted">{STATUS_LABEL[detail.status] ?? ""}</div>}
        {ready && (
          <PlatePanel
            plates={plates}
            selected={selected}
            original={original}
            result={resultIsLatest ? result : null}
            onSelect={setSelected}
            onZoom={(i) => {
              viewer.current?.zoomTo(plates[i].quad);
              if (narrow) setPanel(null);
            }}
            onToggle={(i) => editPlate(i, { enabled: !plates[i].enabled })}
            onRemove={removePlate}
          />
        )}
      </div>
      {detail && (
        <>
          <h2>縁として残す幅</h2>
          <input type="range" min={0} max={10} step={0.5} defaultValue={detail.margin} key={detail.id}
            onChange={(e) => setMargin(Number(e.target.value))} />
          <div className="muted">プレート高さの {detail.margin}%</div>
        </>
      )}
    </>
  );

  const help = coarse ? (
    <div className="keys">
      <div>1本指: 移動　2本指: 拡大縮小</div>
      <div>枠をタップ: 選択（ダブルタップで拡大）</div>
      <div>選んだ枠の角の丸をドラッグ: 四隅を修正</div>
      <div>点線（除外された候補）をタップ: 対象にする</div>
    </div>
  ) : (
    <div className="keys">
      <div>ホイール: 拡大縮小　ドラッグ: 移動</div>
      <div>角の丸をドラッグ: 四隅を修正</div>
      <div>枠をクリック: 選択（ダブルクリックで拡大）</div>
      <div>点線（除外された候補）をクリック: 対象にする</div>
      <div><kbd>X</kbd> 選択中を対象外/対象に　<kbd>Delete</kbd> 削除</div>
      <div><kbd>N</kbd>/<kbd>P</kbd> 次・前の写真　<kbd>Esc</kbd> 追加を取り消し</div>
    </div>
  );

  const stage = (
    <main
      className="stage"
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        upload([...e.dataTransfer.files]);
      }}
    >
      <Viewer
        ref={viewer}
        original={original}
        result={result}
        mode={mode}
        plates={plates}
        selected={selected}
        tool={tool}
        status={status}
        onSelect={setSelected}
        onEnable={(i) => editPlate(i, { enabled: true })}
        onMoveCorner={(i, quad) => editPlate(i, { quad, source: plates[i].source === "auto" ? "auto-edited" : plates[i].source })}
        onToolDone={onToolDone}
        onCancelTool={() => setTool(null)}
      />
      {narrow && original && <div className="floating">{viewControls}</div>}
      {(dragOver || !files.length) && (
        <div className={`drop ${dragOver ? "over" : ""}`}>
          <div>{coarse || narrow ? `「${narrow ? "＋写真" : "写真を追加"}」から写真を選んでください` : "ここに写真をドロップ"}</div>
          <div className="muted small">JPEG・PNG・WebP・TIFF。複数まとめて追加できます</div>
        </div>
      )}
    </main>
  );

  const header = (
    <header>
      {!narrow && <h1>ナンバー消去</h1>}
      <button onClick={() => fileInput.current?.click()}>{narrow ? "＋写真" : "写真を追加"}</button>
      {fileInputEl}
      <span className="spacer" />
      {!narrow && <label className="muted">保存形式</label>}
      <select value={format} onChange={(e) => setFormat(e.target.value as SaveFormat)} aria-label="保存形式">
        <option value="jpeg">JPEG</option>
        <option value="webp">WebP</option>
      </select>
      <DownloadButton ref={downloadOne} target={ready ? currentId : null} format={format}>
        {narrow ? "保存" : <>この写真をダウンロード <kbd>D</kbd></>}
      </DownloadButton>
      <DownloadButton target={files.some((f) => f.width != null) ? "all" : null} format={format} className="primary">
        {narrow ? "全部" : "すべてダウンロード（ZIP）"}
      </DownloadButton>
    </header>
  );

  if (narrow) {
    // 写真を画面の大部分に出し、一覧とナンバーは下のタブで切り替える（開いているタブを押すと閉じる）
    const tab = (p: Exclude<Panel, null>, label: string) => (
      <button className={panel === p ? "on" : ""} onClick={() => setPanel(panel === p ? null : p)}>{label}</button>
    );
    return (
      <div className="app narrow">
        {header}
        {stage}
        <nav className="tabs">
          {tab("files", `写真 ${files.length}`)}
          {tab("plates", `ナンバー ${ready ? plates.filter((p) => p.enabled).length : "-"}`)}
          {panel && <button className="close" onClick={() => setPanel(null)} aria-label="閉じる">▼</button>}
        </nav>
        {panel && (
          <section className="sheet">
            {panel === "files" ? (
              <FileList
                files={files}
                currentId={currentId}
                onSelect={(id) => {
                  setCurrentId(id);
                  setPanel(null); // 選んだ写真を広く見せる
                }}
                onRemove={remove}
              />
            ) : (
              <div className="sheet-body">
                {plateSection}
                <h2>操作</h2>
                {help}
              </div>
            )}
          </section>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      {header}
      <aside className="files">
        <FileList files={files} currentId={currentId} onSelect={setCurrentId} onRemove={remove} />
      </aside>
      {stage}
      <aside className="side">
        <h2>表示</h2>
        {viewControls}
        <h2>ナンバー</h2>
        {plateSection}
        <h2>操作</h2>
        {help}
      </aside>
    </div>
  );
}
