"""人手の注釈（.regression/annotations.json）を正解として、自動処理の精度を測る。

uv run evaluate.py                 # 集計と、見落とし・誤検出・四隅のずれが大きいものの一覧
uv run evaluate.py --save 名前     # 結果を .regression/eval_名前.json に保存
uv run evaluate.py --compare 名前  # 保存した結果と比べて、良くなった・悪くなったナンバーを表示

- 正解: 注釈のナンバーのうち「対象外」でないもの。写真の端に接するもの（切れたナンバー）は
  手動で指定したときだけ処理する方針なので、正解から外して「手動のみ」として別に数える
- 検出: 自動処理の四角形が正解と重なる（IoU 0.3 以上）
- 見落とし: どの自動処理とも重ならない正解
- 誤検出: どの注釈（対象外を含む）とも重ならない自動処理。対象外と重なったものは数えない
- 四隅のずれ: 対応する四隅の距離の最大値を、正解のナンバー幅に対する % で表す
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

import plate

ANNOTATIONS = os.path.join(".regression", "annotations.json")
SRC = "sample_picture"
IOU_MATCH = 0.3
EDGE = 3  # 写真の端からこの px 以内に角がある正解は「端で切れている」とみなす


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = plate.order_quad(a), plate.order_quad(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def corner_error(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(最大の四隅のずれ px, その正解の幅に対する %)"""
    pred, gt = plate.order_quad(pred), plate.order_quad(gt)
    err = float(np.max(np.linalg.norm(pred - gt, axis=1)))
    width = float(np.linalg.norm(gt[1] - gt[0]) + np.linalg.norm(gt[2] - gt[3])) / 2
    return err, 100 * err / max(width, 1.0)


def evaluate_image(name: str, ann: dict) -> dict:
    img, _ = plate.load_image(os.path.join(SRC, name))
    preds = [np.float32(c.quad) for c in plate.find_plates(img) if c.ok]
    H, W = img.shape[:2]
    gts = []
    for p in ann["plates"]:
        q = np.float32(p["quad"])
        edge = q.min() <= EDGE or (q[:, 0] >= W - 1 - EDGE).any() or (q[:, 1] >= H - 1 - EDGE).any()
        gts.append((q, {**p, "ignore": p["ignore"] or bool(edge), "edge": bool(edge) and not p["ignore"]}))

    pairs = sorted(((iou(pq, gq), i, j) for i, pq in enumerate(preds) for j, (gq, _) in enumerate(gts)), reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if v < IOU_MATCH or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, v))

    result = {"detected": [], "missed": [], "false_pos": [], "ignored_hits": 0, "edge_only": sum(m["edge"] for _, m in gts)}
    for i, j, v in matches:
        gq, meta = gts[j]
        if meta["ignore"]:
            result["ignored_hits"] += 1
            continue
        px, pct = corner_error(preds[i], gq)
        result["detected"].append({"gt": j, "iou": round(v, 3), "err_px": round(px, 1), "err_pct": round(pct, 2), "kind": meta["kind"], "center": gq.mean(axis=0).round().tolist()})
    for j, (gq, meta) in enumerate(gts):
        if j not in used_g and not meta["ignore"]:
            result["missed"].append({"gt": j, "kind": meta["kind"], "center": gq.mean(axis=0).round().tolist(), "width": round(float(np.linalg.norm(gq[1] - gq[0])))})
    for i, pq in enumerate(preds):
        if i not in used_p:
            result["false_pos"].append({"center": pq.mean(axis=0).round().tolist()})
    return result


def summarize(results: dict) -> dict:
    det = [d for r in results.values() for d in r["detected"]]
    n_gt = len(det) + sum(len(r["missed"]) for r in results.values())
    errs = np.array([d["err_pct"] for d in det]) if det else np.zeros(1)
    return {
        "targets": n_gt,
        "detected": len(det),
        "missed": n_gt - len(det),
        "false_pos": sum(len(r["false_pos"]) for r in results.values()),
        "edge_only": sum(r.get("edge_only", 0) for r in results.values()),
        "recall": round(len(det) / max(n_gt, 1), 4),
        "err_pct_median": round(float(np.median(errs)), 2),
        "err_pct_p90": round(float(np.percentile(errs, 90)), 2),
        "err_pct_max": round(float(errs.max()), 2),
        "within_3pct": round(float((errs <= 3).mean()), 4),
    }


def print_report(results: dict, s: dict) -> None:
    print(f"正解 {s['targets']} 件 / 検出 {s['detected']} 件（再現率 {100 * s['recall']:.1f}%）/ 見落とし {s['missed']} 件 / 誤検出 {s['false_pos']} 件"
          f"（ほかに写真の端で切れた手動のみのナンバー {s.get('edge_only', 0)} 件）")
    print(f"四隅のずれ（ナンバー幅に対する %）: 中央値 {s['err_pct_median']}%  90%点 {s['err_pct_p90']}%  最大 {s['err_pct_max']}%  3% 以内 {100 * s['within_3pct']:.0f}%")
    print("\n見落とし:")
    for name, r in results.items():
        for m in r["missed"]:
            print(f"  {name} 中心 {m['center']} 幅 {m['width']}px {'' if m['kind'] == 'normal' else '(' + m['kind'] + ')'}")
    print("\n誤検出:")
    for name, r in results.items():
        for f in r["false_pos"]:
            print(f"  {name} 中心 {f['center']}")
    worst = sorted(((d["err_pct"], d["err_px"], name, d["center"]) for name, r in results.items() for d in r["detected"]), reverse=True)[:10]
    print("\n四隅のずれが大きいもの（上位10件）:")
    for pct, px, name, c in worst:
        print(f"  {pct:5.1f}% ({px:.0f}px)  {name} 中心 {c}")


def compare(results: dict, old: dict) -> None:
    print("\n前回との比較:")
    for name in sorted(set(results) | set(old)):
        a, b = old.get(name), results.get(name)
        if a is None or b is None:
            continue
        ea = {d["gt"]: d["err_pct"] for d in a["detected"]}
        eb = {d["gt"]: d["err_pct"] for d in b["detected"]}
        for g in sorted(set(ea) | set(eb)):
            if g not in eb:
                print(f"  悪化  {name} 正解#{g + 1} が見落としになった")
            elif g not in ea:
                print(f"  改善  {name} 正解#{g + 1} を検出するようになった")
            elif abs(eb[g] - ea[g]) >= 1.0:
                print(f"  {'改善' if eb[g] < ea[g] else '悪化'}  {name} 正解#{g + 1} 四隅のずれ {ea[g]}% → {eb[g]}%")
        if len(b["false_pos"]) != len(a["false_pos"]):
            print(f"  {'改善' if len(b['false_pos']) < len(a['false_pos']) else '悪化'}  {name} 誤検出 {len(a['false_pos'])} → {len(b['false_pos'])} 件")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", help="結果を .regression/eval_<名前>.json に保存する")
    ap.add_argument("--compare", help="保存した結果 .regression/eval_<名前>.json と比べる")
    args = ap.parse_args()

    with open(ANNOTATIONS, encoding="utf-8") as f:
        anns = {n: a for n, a in json.load(f)["images"].items() if a.get("done")}
    t = time.time()
    results = {name: evaluate_image(name, ann) for name, ann in sorted(anns.items())}
    s = summarize(results)
    print(f"{len(anns)} 枚を評価（{time.time() - t:.0f} 秒）\n")
    print_report(results, s)
    if args.compare:
        with open(os.path.join(".regression", f"eval_{args.compare}.json"), encoding="utf-8") as f:
            old = json.load(f)
        o = old["summary"]
        print(f"\n前回 ({args.compare}): 検出 {o['detected']}/{o['targets']}  誤検出 {o['false_pos']}  ずれ中央値 {o['err_pct_median']}%  90%点 {o['err_pct_p90']}%")
        compare(results, old["images"])
    if args.save:
        with open(os.path.join(".regression", f"eval_{args.save}.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": s, "images": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
