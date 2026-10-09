# -*- coding: utf-8 -*-
"""
calibrate.py —— 离线阈值校准工具

作用：把 main.py 里的 WHOLE_THRESHOLD / PIECE_THRESHOLD 从"拍脑袋"变成"量出来的"。

它做三件事：
  【1】模板自检：每张模板和它自己比，分数应该≈1.0。用来验证环境正常
       （模板没损坏、灰度转换路径一致）。如果不接近 1.0，说明读取环节有问题，
       后面的数字都不用看了。
  【2】交叉干扰：4 张全图模板两两比、16 张碎片模板两两比，得到"错误匹配的
       静态上界"——它决定了阈值至少要高于多少。
  【3】合成退化扫描：把模板按"游戏窗口分辨率 != 基准分辨率 + 裁剪位置有残差
       + 屏幕噪点"的方式退化一遍，统计正确匹配的最低分与错误匹配的最高分，
       按规则（正确最小值 - 0.02 / 错误最大值 + 0.02）给出建议阈值。

如果根目录存在 test/ 目录，还会读入真实截图做统计（命名约定见下），
真实数据的结论优先于合成数据。

test/ 命名约定：
    {idx}_{pos}.png   某个碎片槽位的截图。idx = 当前大指纹编号 0-3，pos = 槽位 0-7。
    whole_{idx}.png   （可选）指纹大图截图，idx = 该图属于哪一组 0-3。

用法：
    python calibrate.py                     # 用默认的 test/ 目录（存在才读）
    python calibrate.py --test-dir shots
    python calibrate.py --piece-margin 3
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

import main as app
import pic


# ==========================================================================
# 合成退化：模拟"窗口分辨率不同 + 位置有残差 + 屏幕噪点"
# ==========================================================================

def simulate(img, win_w, win_h, shift_px, noise_sigma, seed=0):
    """把一张基准分辨率的模板，退化成一个"真实截屏里大概会是什么样"。

    win_w/win_h : 游戏窗口的实际像素尺寸（决定缩放比）
    shift_px    : 裁剪位置的残余偏移（matchTemplate 的滑动余量会吸收掉一部分）
    noise_sigma : 灰度噪声标准差（代表红绿噪点、视频压缩、抗锯齿等）
    """
    out = img

    factor_x = win_w / float(pic.BASE_WIDTH)
    factor_y = win_h / float(pic.BASE_HEIGHT)
    if abs(factor_x - 1.0) > 1e-9 or abs(factor_y - 1.0) > 1e-9:
        small_w = max(1, int(round(out.shape[1] * factor_x)))
        small_h = max(1, int(round(out.shape[0] * factor_y)))
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)
        # 与 pic._resize_to_base 保持同一套插值策略
        if factor_x >= 1.0 and factor_y >= 1.0:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        out = cv2.resize(small, (out.shape[1], out.shape[0]), interpolation=interp)

    if shift_px:
        matrix = np.float32([[1, 0, shift_px], [0, 1, shift_px]])
        out = cv2.warpAffine(out, matrix, (out.shape[1], out.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)

    if noise_sigma:
        rng = np.random.default_rng(seed)
        out = np.clip(out.astype(np.float32) + rng.normal(0, noise_sigma, out.shape),
                      0, 255).astype(np.uint8)

    return out


# 场景表：(窗口宽, 窗口高, 残余偏移px, 噪点sigma, 说明)
SCENARIOS = [
    (2560, 1440, 0, 0,  "原生 1440p 全屏（与模板同源，理想情况）"),
    (2560, 1440, 1, 5,  "1440p + 1px 残差 + 轻噪点"),
    (1920, 1080, 2, 10, "1080p 窗口 + 2px 残差 + 噪点（常见情况）"),
    (1600, 900,  3, 15, "1600x900 窗口 + 3px 残差 + 噪点"),
    (1280, 720,  3, 15, "720p 窗口 + 3px 残差 + 噪点（最恶劣）"),
]


# ==========================================================================
# 输出小工具
# ==========================================================================

def stat_line(name, correct, wrong):
    c = np.asarray(correct, dtype=float)
    w = np.asarray(wrong, dtype=float) if len(wrong) else np.asarray([-1.0])
    print(f"    {name:<6} 正确 min={c.min():+.4f} mean={c.mean():+.4f} | "
          f"错误 max={w.max():+.4f} mean={w.mean():+.4f} | 间隔={c.min() - w.max():+.4f}")
    return float(c.min()), float(w.max())


def recommend(correct, wrong, label):
    """按"正确最小值 - 0.02（且需高于 错误最大值 + 0.02）"给出建议阈值。"""
    c = np.asarray(correct, dtype=float)
    w = np.asarray(wrong, dtype=float) if len(wrong) else np.asarray([-1.0])
    cmin, wmax = float(c.min()), float(w.max())
    gap = cmin - wmax

    print(f"    {label}: 正确最低 {cmin:+.4f} / 错误最高 {wmax:+.4f} / 间隔 {gap:+.4f}")
    if gap <= 0:
        print("      !! 正确与错误分数重叠，单一阈值无法分开。"
              "请先排查：窗口是否被拉伸、是否落在非 16:9、模板是否过期。")
        return None

    hi = cmin - 0.02          # 保召回：贴着正确样本的下界往下让 0.02
    lo = wmax + 0.02          # 保精度：贴着错误样本的上界往上让 0.02
    if hi > lo:
        print(f"      建议阈值 = 正确最小值 - 0.02 = {hi:.4f}（同时高于 错误最大值 + 0.02 = {lo:.4f}，安全）")
    else:
        print(f"      建议阈值 = 正确最小值 - 0.02 = {hi:.4f}"
              f"（低于 错误最大值 + 0.02 = {lo:.4f}，两类样本靠得近，建议先改善对齐再调阈值）")
    return hi


# ==========================================================================
# 【1】模板自检
# ==========================================================================

def self_check(whole, pieces):
    print("\n【1】模板自检（模板与自身匹配，理想值 1.0）")
    worst = 1.0
    worst_name = ""
    for i in sorted(whole):
        s = app._match_score(whole[i], whole[i], app.WHOLE_SEARCH_MARGIN)
        if s < worst:
            worst, worst_name = s, f"全图 {i}"
        print(f"    全图 {i}: {s:+.6f}")
    for i in sorted(pieces):
        for j, img in enumerate(pieces[i]):
            s = app._match_score(img, img, app.PIECE_SEARCH_MARGIN)
            if s < worst:
                worst, worst_name = s, f"碎片 {i}_{j}"
            print(f"    碎片 {i}_{j}: {s:+.6f}")

    if worst >= 0.9999:
        print(f"    => 通过（最低 {worst:.6f}）。模板可读、灰度路径一致，环境正常。")
        return True
    print(f"    => 注意：最低分出现在 {worst_name}（{worst:+.6f}），低于 1.0。"
          f"常见原因是模板本身分辨率就不是 141x141 / 460x675，或图像被重新压缩过。")
    return False


# ==========================================================================
# 【2】交叉干扰
# ==========================================================================

def cross_check(whole, pieces):
    print("\n【2】交叉干扰（不同模板之间的分数，代表「错误匹配」的静态上界）")

    whole_wrong = []
    for i in sorted(whole):
        for k in sorted(whole):
            if i != k:
                whole_wrong.append(app._match_score(whole[i], whole[k], app.WHOLE_SEARCH_MARGIN))
    w = np.asarray(whole_wrong)
    print(f"    全图模板两两: 共 {w.size} 对，max={w.max():+.4f} mean={w.mean():+.4f}")
    print(f"      => 全图阈值必须高于 {w.max():+.4f}")

    piece_wrong = []
    for i in sorted(pieces):
        for j in range(4):
            for k in sorted(pieces):
                for m in range(4):
                    if (i, j) != (k, m):
                        piece_wrong.append(app._match_score(pieces[k][m], pieces[i][j],
                                                            app.PIECE_SEARCH_MARGIN))
    p = np.asarray(piece_wrong)
    print(f"    碎片模板两两: 共 {p.size} 对，max={p.max():+.4f} mean={p.mean():+.4f}")
    print(f"      => 碎片阈值必须高于 {p.max():+.4f}")

    return float(w.max()), float(p.max())


# ==========================================================================
# 【3】合成退化扫描
# ==========================================================================

def synthetic_check(whole, pieces):
    print("\n【3】合成退化扫描（模拟窗口分辨率不同 + 位置残差 + 噪点）")
    print(f"    全图滑动余量 {app.WHOLE_SEARCH_MARGIN}px / 碎片滑动余量 {app.PIECE_SEARCH_MARGIN}px")

    whole_recs, piece_recs = [], []

    for win_w, win_h, shift, sigma, desc in SCENARIOS:
        print(f"\n  -- {desc} --")

        # 全图：正确 = 自己比自己；错误 = 别的组的截图拿去比本组模板
        correct, wrong = [], []
        for i in sorted(whole):
            sim_i = simulate(whole[i], win_w, win_h, shift, sigma, seed=i)
            correct.append(app._match_score(sim_i, whole[i], app.WHOLE_SEARCH_MARGIN))
            for k in sorted(whole):
                if k != i:
                    sim_k = simulate(whole[k], win_w, win_h, shift, sigma, seed=k)
                    wrong.append(app._match_score(sim_k, whole[i], app.WHOLE_SEARCH_MARGIN))
        stat_line("全图", correct, wrong)
        rec = recommend(correct, wrong, "全图建议")
        if rec is not None:
            whole_recs.append(rec)

        # 碎片：正确 = 自己的截图比自己的模板；
        #       错误 = 任意别的碎片（同组或跨组）的截图，拿去比本片模板
        correct, wrong = [], []
        for i in sorted(pieces):
            for j in range(4):
                sim_ij = simulate(pieces[i][j], win_w, win_h, shift, sigma, seed=i * 4 + j)
                correct.append(app._match_score(sim_ij, pieces[i][j], app.PIECE_SEARCH_MARGIN))
                for k in sorted(pieces):
                    for m in range(4):
                        if (i, j) == (k, m):
                            continue
                        sim_km = simulate(pieces[k][m], win_w, win_h, shift, sigma, seed=k * 4 + m)
                        wrong.append(app._match_score(sim_km, pieces[i][j], app.PIECE_SEARCH_MARGIN))
        stat_line("碎片", correct, wrong)
        rec = recommend(correct, wrong, "碎片建议")
        if rec is not None:
            piece_recs.append(rec)

    whole_rec = min(whole_recs) if whole_recs else None
    piece_rec = min(piece_recs) if piece_recs else None
    print("\n  合成场景汇总（取各场景建议值里最保守的那个，保证最恶劣场景也能召回）:")
    print(f"    WHOLE_THRESHOLD = {whole_rec:.4f}" if whole_rec is not None else "    WHOLE_THRESHOLD = 无法标定")
    print(f"    PIECE_THRESHOLD = {piece_rec:.4f}" if piece_rec is not None else "    PIECE_THRESHOLD = 无法标定")
    return whole_rec, piece_rec


# ==========================================================================
# 【4】真实截图统计（test/ 目录存在时）
# ==========================================================================

PIECE_NAME_RE = re.compile(r"^(\d+)_(\d+)$")
WHOLE_NAME_RE = re.compile(r"^whole_(\d+)$", re.IGNORECASE)


def _load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"      !! 无法读取: {path}")
    return img


def _fit(img, expect_w, expect_h):
    """把截图缩放成期望尺寸（截图分辨率与基准不同时）。"""
    if img is None:
        return None
    if img.shape[1] == expect_w and img.shape[0] == expect_h:
        return img
    interp = cv2.INTER_AREA if (img.shape[1] >= expect_w and img.shape[0] >= expect_h) else cv2.INTER_CUBIC
    return cv2.resize(img, (expect_w, expect_h), interpolation=interp)


def real_check(test_dir, whole, pieces):
    print(f"\n【4】真实截图统计（目录: {test_dir}）")
    if not os.path.isdir(test_dir):
        print("    目录不存在，跳过。想用真实数据标定时，把截图按 "
              "{idx}_{pos}.png 命名放进去即可。")
        return None, None

    names = sorted(os.listdir(test_dir))
    piece_files, whole_files = [], []
    for name in names:
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        m = PIECE_NAME_RE.match(stem)
        if m:
            piece_files.append((int(m.group(1)), int(m.group(2)), os.path.join(test_dir, name)))
            continue
        m = WHOLE_NAME_RE.match(stem)
        if m:
            whole_files.append((int(m.group(1)), os.path.join(test_dir, name)))

    whole_rec = piece_rec = None

    # ---- 全图 ----
    if whole_files:
        print("\n  -- 全图截图 --")
        correct, wrong = [], []
        for idx, path in whole_files:
            img = _fit(_load_gray(path), pic.WHOLE_WIDTH, pic.WHOLE_HEIGHT)
            if img is None:
                continue
            scores = app.whole_scores(img, whole, app.WHOLE_SEARCH_MARGIN)
            by_index = dict(scores)
            best_index, best_score = max(scores, key=lambda t: t[1])
            tag = "一致" if best_index == idx else f"!! 标注是 {idx}，最高分却给了 {best_index}"
            print(f"    {os.path.basename(path)}: 标注={idx} 最高={best_index}({best_score:+.4f}) {tag}")
            correct.append(by_index[idx])
            wrong.extend(s for i, s in scores if i != idx)
        stat_line("全图", correct, wrong)
        whole_rec = recommend(correct, wrong, "全图建议")
    else:
        print("\n  -- 全图截图 -- 没有 whole_{idx}.png，跳过")

    # ---- 碎片 ----
    if piece_files:
        print("\n  -- 碎片截图（{idx}_{pos}.png）--")
        print("    说明：文件名只标注了「属于哪一组」，没有标注是组内第几片，"
              "因此用 16 张模板里的最高分作为自标注（这也正是运行时的判定方式）。")
        correct, wrong = [], []
        for idx, pos, path in piece_files:
            img = _fit(_load_gray(path), pic.PIECE_WIDTH, pic.PIECE_HEIGHT)
            if img is None:
                continue
            best = (-1.0, None)
            all_scores = []
            for k in sorted(pieces):
                for m in range(4):
                    s = app._match_score(img, pieces[k][m], app.PIECE_SEARCH_MARGIN)
                    all_scores.append(((k, m), s))
                    if s > best[0]:
                        best = (s, (k, m))
            best_score, best_key = best
            group_ok = best_key[0] == idx
            print(f"    {os.path.basename(path)}: 标注组={idx} 最高={best_key}({best_score:+.4f}) "
                  f"{'一致' if group_ok else '!! 组不一致，建议人工确认这张截图'}")
            correct.append(best_score)
            wrong.extend(s for key, s in all_scores if key != best_key)
        stat_line("碎片", correct, wrong)
        piece_rec = recommend(correct, wrong, "碎片建议")
    else:
        print("\n  -- 碎片截图 -- 没有 {idx}_{pos}.png，跳过")

    return whole_rec, piece_rec


# ==========================================================================
# 主流程
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="GTAOL 指纹识别阈值校准工具")
    parser.add_argument("--test-dir", default="test",
                        help="真实截图目录（默认 test；不存在则只用合成数据标定）")
    parser.add_argument("--whole-margin", type=int, default=None,
                        help="全图匹配滑动余量，默认沿用 main.WHOLE_SEARCH_MARGIN")
    parser.add_argument("--piece-margin", type=int, default=None,
                        help="碎片匹配滑动余量，默认沿用 main.PIECE_SEARCH_MARGIN")
    args = parser.parse_args()

    if args.whole_margin is not None:
        app.WHOLE_SEARCH_MARGIN = args.whole_margin
    if args.piece_margin is not None:
        app.PIECE_SEARCH_MARGIN = args.piece_margin

    print("=" * 78)
    print("GTAOL 指纹识别 —— 阈值校准")
    print("=" * 78)
    print(f"基准分辨率 {pic.BASE_WIDTH}x{pic.BASE_HEIGHT}；"
          f"全图裁剪 {pic.WHOLE_WIDTH}x{pic.WHOLE_HEIGHT}；"
          f"碎片槽位 {pic.PIECE_WIDTH}x{pic.PIECE_HEIGHT}")
    print(f"当前阈值: WHOLE_THRESHOLD={app.WHOLE_THRESHOLD:.2f} "
          f"PIECE_THRESHOLD={app.PIECE_THRESHOLD:.2f}")
    print(f"滑动余量: 全图 {app.WHOLE_SEARCH_MARGIN}px / 碎片 {app.PIECE_SEARCH_MARGIN}px")

    print("\n【0】加载模板")
    try:
        whole, pieces = app.load_all_templates()
    except Exception as exc:
        print(f"    模板加载失败: {exc}")
        return 1
    print(f"    加载成功: 全图 {len(whole)} 组，碎片 {len(pieces)} 组 "
          f"（合计 {len(whole) + sum(len(g) for g in pieces.values())} 张）")

    self_check(whole, pieces)
    cross_check(whole, pieces)
    syn_whole, syn_piece = synthetic_check(whole, pieces)
    real_whole, real_piece = real_check(args.test_dir, whole, pieces)

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)

    def fmt(value):
        return f"{value:.4f}" if value is not None else "(无法标定)"

    print(f"  合成数据建议: WHOLE_THRESHOLD = {fmt(syn_whole)}   "
          f"PIECE_THRESHOLD = {fmt(syn_piece)}")

    if real_whole is not None or real_piece is not None:
        print(f"  真实数据建议: WHOLE_THRESHOLD = {fmt(real_whole)}   "
              f"PIECE_THRESHOLD = {fmt(real_piece)}")
        print("  => 真实数据优先。把上面的值填回 main.py 顶部的 "
              "WHOLE_THRESHOLD / PIECE_THRESHOLD。")
    else:
        print("  => 目前只有合成数据。跑几把真实对局，把截图放进 test/ 再跑一次，"
              "结果会更贴近你的机器。")
    print("  => 改完阈值后建议复跑一次本脚本，确认间隔没有变小。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
