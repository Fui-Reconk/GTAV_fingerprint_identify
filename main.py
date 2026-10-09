# -*- coding: utf-8 -*-
"""
main.py —— GTAOL 指纹破解助手（图像识别层重构版）

相对旧版的改动集中在"识别"这一层，业务层（GUI / 日志 / 快捷键 / control）保持不变：

1. 模板预加载：启动时一次性把 4 张全图模板 + 16 张碎片模板读进内存，
   不再像旧代码那样每次匹配都 cv2.imread 读一次磁盘。
2. 匹配算法：SSIM（skimage，逐像素、对亮度/对比度敏感）
   → cv2.matchTemplate + TM_CCOEFF_NORMED（归一化互相关，对亮度/对比度
   完全不敏感，对红绿噪点更鲁棒，而且底层是 C++ 实现，快几十倍）。
3. 只截一次图：整个流程只调用一次 get_game_window_img()，全图与 8 个碎片
   都从这一张基准图里裁剪。
4. 碎片匹配改为"在未标记模板里取最高分"（旧版是"先到先得"）：
   每个模板仍最多被选中一次（marked 互斥），但不再因为遍历顺序而误判。

阈值不是拍脑袋定的，标定过程见 calibrate.py；REFACTOR_NOTES.md 里有完整推导。
"""

import json
import os
import re
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import keyboard
import numpy as np
import pydirectinput
import winsound

from pic import (
    BASE_HEIGHT,
    BASE_WIDTH,
    WHOLE_HEIGHT,
    WHOLE_WIDTH,
    PIECE_HEIGHT,
    PIECE_WIDTH,
    get_fingerprint_pics,
    get_game_window_img,
    get_last_window_info,
    get_whole_fingerprint,
    release_resources,
)


# ==========================================================================
# 一、可调参数
# ==========================================================================

def _app_dir():
    """程序所在目录。

    打包成 exe（PyInstaller）之后 __file__ 指向临时解包目录，因此冻结状态下
    必须用可执行文件所在目录，config.json 和 pic/ 才找得到。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _app_dir()
# 模板根目录: pic/fingerprint{idx}/fingerprint{idx}[_{piece}].png
TEMPLATE_ROOT = os.path.join(APP_DIR, "pic")
# 界面设置持久化到这里
CONFIG_PATH = os.path.join(APP_DIR, "config.json")

# --- 阈值（由 calibrate.py 实测得出，不是拍脑袋，推导见 REFACTOR_NOTES.md）---
# 全图：合成退化扫描里正确匹配的最低分是 0.8507（720p 窗口 + 3px 残差 + 噪点），
#       按 calibrate.py 的规则取 0.8507 - 0.02 ≈ 0.83；
#       而不同指纹模板之间最高只有 0.276，0.83 离错误上界有 0.55 的安全距离。
WHOLE_THRESHOLD = 0.83
# 碎片：同样场景下正确匹配最低 0.8363，取 0.8363 - 0.02 ≈ 0.81；
#       「别的碎片」最高只有 0.353，0.81 离错误上界还有 0.46 的安全距离。
PIECE_THRESHOLD = 0.81

# --- 匹配时允许的滑动余量（像素）------------------------------------------
# 模板与裁剪框同为 141x141 / 460x675。若直接用整块模板去比，matchTemplate
# 只会得到 1x1 的结果（等价于逐像素相关），裁剪位置差 1px 分数就从 1.0
# 掉到 0.87，差 3px 掉到 0.52 —— 实测过，这是迁到"任意分辨率 + 窗口可移动"
# 之后最大的准确率杀手。
# 因此改用"模板内部区域"去比整块裁剪框，让 matchTemplate 自己找最佳位置，
# 相当于获得了 ±margin 像素的平移容差。实测 margin=4 时：
#   1080p 窗口下正确分从 0.44 拉回 0.91，错误分仅从 0.276 升到 0.333。
# margin 继续加大收益递减且错误分会慢慢升高，4 是实测的平衡点。
WHOLE_SEARCH_MARGIN = 4
PIECE_SEARCH_MARGIN = 4

# 按下开始快捷键后，留给游戏把指纹界面画出来的时间（等价旧版 get_whole_fingerprint 里的 sleep(3)）
UI_SETTLE_DELAY = 3.0

# --- 模板容器（启动时由 load_all_templates() 填充）-------------------------
WHOLE_TEMPLATES = {}   # {index: 灰度全图模板 ndarray}
PIECE_TEMPLATES = {}   # {index: [4 个灰度碎片模板 ndarray]}

# 碎片模板的互斥标记：每个模板最多被选中一次（保留旧代码的全局引用方式）
marked = [False] * 4


# ==========================================================================
# 二、模板预加载
# ==========================================================================

def load_all_templates(root=TEMPLATE_ROOT):
    """启动时一次性把所有模板读入内存。

    返回 (whole_templates, piece_templates)：
        whole_templates = {index: 灰度全图}
        piece_templates = {index: [4 个灰度碎片]}

    模板缺失、读不出来、或者尺寸与坐标常量对不上时，直接抛 RuntimeError，
    错误消息里点名是哪个文件、期望多大、实际多大。
    """
    whole_templates = {}
    piece_templates = {}
    missing = []

    for idx in range(4):
        whole_path = os.path.join(root, f"fingerprint{idx}", f"fingerprint{idx}.png")
        whole_img = cv2.imread(whole_path, cv2.IMREAD_GRAYSCALE)
        if whole_img is None:
            missing.append(whole_path)
        else:
            whole_templates[idx] = whole_img

        group = []
        for piece in range(4):
            piece_path = os.path.join(root, f"fingerprint{idx}", f"fingerprint{idx}_{piece}.png")
            piece_img = cv2.imread(piece_path, cv2.IMREAD_GRAYSCALE)
            if piece_img is None:
                missing.append(piece_path)
            group.append(piece_img)
        piece_templates[idx] = group

    if missing:
        raise RuntimeError(
            "模板文件缺失或无法读取（共 %d 个）:\n  %s\n"
            "请确认模板目录结构与原版一致: %s/fingerprint{0-3}/fingerprint{0-3}[_{0-3}].png"
            % (len(missing), "\n  ".join(missing), root)
        )

    # ---- 尺寸校验：模板必须与坐标常量推出的裁剪框一致 ----
    for idx, img in whole_templates.items():
        if img.shape[1] != WHOLE_WIDTH or img.shape[0] != WHOLE_HEIGHT:
            raise RuntimeError(
                f"全图模板尺寸不匹配: fingerprint{idx}.png 实际 {img.shape[1]}x{img.shape[0]}，"
                f"但 START_POINT/END_POINT 决定裁剪框应为 {WHOLE_WIDTH}x{WHOLE_HEIGHT}。"
                f"模板必须基于 {BASE_WIDTH}x{BASE_HEIGHT} 基准分辨率采集。"
            )
    for idx, group in piece_templates.items():
        for piece, img in enumerate(group):
            if img.shape[1] != PIECE_WIDTH or img.shape[0] != PIECE_HEIGHT:
                raise RuntimeError(
                    f"碎片模板尺寸不匹配: fingerprint{idx}_{piece}.png 实际 "
                    f"{img.shape[1]}x{img.shape[0]}，但 TOP_LEFT/BOTTOM_RIGHT 决定槽位应为 "
                    f"{PIECE_WIDTH}x{PIECE_HEIGHT}。"
                )
            if img.shape[0] <= 2 * PIECE_SEARCH_MARGIN or img.shape[1] <= 2 * PIECE_SEARCH_MARGIN:
                raise RuntimeError(
                    f"碎片模板 fingerprint{idx}_{piece}.png 太小（{img.shape[1]}x{img.shape[0]}），"
                    f"留不出 {PIECE_SEARCH_MARGIN}px 的滑动余量，请调小 PIECE_SEARCH_MARGIN。"
                )

    if len(whole_templates) != 4 or len(piece_templates) != 4:
        raise RuntimeError(
            f"模板组数不完整: 全图 {len(whole_templates)} 组 / 碎片 {len(piece_templates)} 组，期望各 4 组。"
        )

    return whole_templates, piece_templates


# ==========================================================================
# 三、匹配核心
# ==========================================================================

def _inner_roi(template_gray, margin):
    """取模板的内部区域。

    用内部区域去比整块裁剪框，matchTemplate 就能在 ±margin 像素范围内
    自行滑动对齐，从而容忍窗口位置/缩放带来的亚像素错位。
    margin=0 时退化为"整块对整块"（结果只有 1x1）。
    """
    if margin <= 0:
        return template_gray
    height, width = template_gray.shape[:2]
    if height <= 2 * margin or width <= 2 * margin:
        raise RuntimeError(
            f"模板 {width}x{height} 不足以留出 {margin}px 的滑动余量，请调小 SEARCH_MARGIN"
        )
    return template_gray[margin:height - margin, margin:width - margin]


def _match_score(crop_gray, template_gray, margin):
    """在 crop_gray 上滑动 template_gray 的内部区域，返回最高相关分。

    使用 TM_CCOEFF_NORMED（归一化互相关）：它先减去均值，
    因此对整体亮度和对比度的线性变化完全不敏感；分数范围约 [-1, 1]。

    返回 -1.0 表示"无法判定"（模板或画面是常数块，归一化相关无定义）。
    """
    if crop_gray is None or template_gray is None:
        return -1.0

    roi = _inner_roi(template_gray, margin)
    crop_h, crop_w = crop_gray.shape[:2]
    roi_h, roi_w = roi.shape[:2]

    if roi_h > crop_h or roi_w > crop_w:
        raise RuntimeError(
            f"尺寸不匹配: 模板内部区 {roi_w}x{roi_h} 大于待匹配画面 {crop_w}x{crop_h}。"
            f"通常是模板不是按 {BASE_WIDTH}x{BASE_HEIGHT} 基准采集的。"
        )
    if float(roi.std()) == 0.0:
        return -1.0

    result = cv2.matchTemplate(crop_gray, roi, cv2.TM_CCOEFF_NORMED)
    finite = result[np.isfinite(result)]
    if finite.size == 0:
        # 画面是常数块（例如整块纯黑）时归一化相关为 NaN
        return -1.0
    return float(finite.max())


def whole_scores(gray, templates=None, margin=WHOLE_SEARCH_MARGIN):
    """对 4 张全图模板打分，返回 [(index, score), ...]（按 index 升序）。

    calibrate.py 直接复用这个函数，保证"标定用的打分"和"运行时用的打分"
    是同一条代码路径。
    """
    templates = templates if templates is not None else WHOLE_TEMPLATES
    if not templates:
        raise RuntimeError("全图模板尚未加载，请先调用 load_all_templates()")
    return [(i, _match_score(gray, templates[i], margin)) for i in sorted(templates)]


def piece_scores(gray, index, templates=None, margin=PIECE_SEARCH_MARGIN, marked_mask=None):
    """对某一组碎片模板打分，返回 [(piece, score), ...]。

    marked_mask 传入时只统计尚未被标记的模板。
    """
    templates = templates if templates is not None else PIECE_TEMPLATES
    group = templates.get(index)
    if not group:
        raise RuntimeError(
            f"未找到指纹 {index} 的碎片模板，请检查 {TEMPLATE_ROOT}/fingerprint{index}/ 目录"
        )
    out = []
    for piece, tmpl in enumerate(group):
        if marked_mask is not None and marked_mask[piece]:
            continue
        out.append((piece, _match_score(gray, tmpl, margin)))
    return out


def Match_whole_pic(fingerprint, templates=None, margin=WHOLE_SEARCH_MARGIN,
                    threshold=WHOLE_THRESHOLD, debug=None):
    """对比 4 张全图模板找到当前是哪一组指纹。

    取分数最高的那组；最高分低于 threshold 时返回 -1（表示"画面上没有指纹"）。
    debug 传入字典时会被填入 scores / best_index / best_score，供上层写日志。
    """
    scores = whole_scores(fingerprint, templates, margin)

    best_index, best_score = -1, -1.0
    for i, score in scores:
        if score > best_score:
            best_index, best_score = i, score
    matched = best_index if best_score >= threshold else -1

    if debug is not None:
        debug["scores"] = scores
        debug["best_index"] = best_index
        debug["best_score"] = best_score
        debug["threshold"] = threshold
        debug["matched"] = matched
    return matched


def Match_piece_pic(pic, index, marked_mask, templates=None, margin=PIECE_SEARCH_MARGIN,
                    threshold=PIECE_THRESHOLD, debug=None):
    """判断某个槽位上的碎片属于 index 组里的哪一片。

    在**尚未被标记**的模板里挑分数最高的那个（而不是像旧版那样"先到先得"），
    分数达到 threshold 才标记该模板并返回 True。每个模板最多被选中一次。
    """
    candidates = piece_scores(pic, index, templates, margin, marked_mask)

    best_piece, best_score = -1, -1.0
    for piece, score in candidates:
        if score > best_score:
            best_piece, best_score = piece, score

    ok = best_piece >= 0 and best_score >= threshold
    if ok:
        marked_mask[best_piece] = True

    if debug is not None:
        debug["scores"] = candidates
        debug["best_index"] = best_piece
        debug["best_score"] = best_score
        debug["threshold"] = threshold
        debug["matched"] = ok
    return ok


# ==========================================================================
# 三点五、配置读写（config.json）
# ==========================================================================

DEFAULT_CONFIG = {
    "start_key": "f11",     # 开始快捷键，取值是 keyboard 库认得的键名
    "end_key": "f12",       # 结束快捷键
    "interval": 0.02,       # 每次模拟按键之间的间隔（秒）
}

# tkinter 的 keysym -> keyboard 库的键名。只有特殊键需要映射，
# 字母/数字直接用 keysym 本身。
# 注意：tkinter 的 keysym 大小写并不统一（Return / Escape / Up / Prior 是大写开头，
# space / Control_L 又不是），所以下面统一用小写作键，查表前先 keysym.lower()。
_KEYSYM_ALIASES = {
    "space": "space",
    "return": "enter", "kp_enter": "enter",
    "escape": "esc",
    "tab": "tab",
    "backspace": "backspace",
    "delete": "delete", "insert": "insert",
    "prior": "page up", "next": "page down",
    "home": "home", "end": "end",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "caps_lock": "caps lock", "scroll_lock": "scroll lock", "num_lock": "num lock",
    "pause": "pause", "print": "print screen", "menu": "menu",
    "control_l": "ctrl", "control_r": "ctrl",
    "shift_l": "shift", "shift_r": "shift",
    "alt_l": "alt", "alt_r": "alt",
    "super_l": "windows", "super_r": "windows",
}


def normalize_keysym(keysym):
    """把 tkinter 的 keysym 转成 keyboard 库能识别的键名。

    不认识的键返回 None，由调用方决定是忽略还是提示。
    """
    lowered = keysym.lower()
    if lowered in _KEYSYM_ALIASES:
        return _KEYSYM_ALIASES[lowered]
    if re.fullmatch(r"f\d{1,2}", lowered):        # F1 ~ F24
        return lowered
    match = re.fullmatch(r"kp_(\d)", lowered)     # 小键盘数字
    if match:
        return match.group(1)
    if len(keysym) == 1 and keysym.isprintable():  # 字母 / 数字 / 符号
        return lowered
    return None


def load_config(path=CONFIG_PATH):
    """读取配置文件。

    返回 (config, warning)：
      - 文件不存在 -> 直接用默认值，warning 为 None（不算异常）
      - 文件损坏或某项非法 -> 该项回退默认值，warning 里说明原因
    """
    config = dict(DEFAULT_CONFIG)
    if not os.path.exists(path):
        return config, None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return config, f"配置文件读取失败（{exc}），已使用默认设置"

    if not isinstance(data, dict):
        return config, "配置文件格式不对（顶层不是 JSON 对象），已使用默认设置"

    problems = []
    for name in ("start_key", "end_key"):
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            config[name] = value.strip().lower()
        elif value is not None:
            problems.append(name)

    if "interval" in data:
        try:
            interval = float(data["interval"])
            if 0 <= interval <= 5:
                config["interval"] = interval
            else:
                problems.append("interval")
        except (TypeError, ValueError):
            problems.append("interval")

    if config["start_key"] == config["end_key"]:
        config["start_key"] = DEFAULT_CONFIG["start_key"]
        config["end_key"] = DEFAULT_CONFIG["end_key"]
        problems.append("开始/结束快捷键相同")

    warning = f"配置项 {'、'.join(problems)} 非法，已改用默认值" if problems else None
    return config, warning


def save_config(config, path=CONFIG_PATH):
    """把配置写回文件。

    先写临时文件再原子替换，避免写到一半崩溃把原配置弄坏。
    写失败时抛出异常，由调用方记录到日志。
    """
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


# ==========================================================================
# 四、业务层（保持原样）
# ==========================================================================

def delete_dir_file(dir_path):
    """
    递归删除文件夹下文件和子文件夹里的文件，不会删除空文件夹
    """
    if not os.path.exists(dir_path):
        return

    if os.path.isfile(dir_path) and os.path.exists(dir_path):
        os.remove(dir_path)  # 删除单个文件
    else:
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            delete_dir_file(os.path.join(dir_path, file_name))

    if os.path.exists(dir_path):
        os.rmdir(dir_path)


def control(command_arr):
    """
    控制键盘操作，输入标记正确选项的布尔数组
    """
    cnt = 0; chosen = 0
    for detect in command_arr:
        if cnt == 4:
            pydirectinput.press("right")
        if detect:
            chosen += 1
            pydirectinput.press("enter")
        if chosen == 4:
            break
        cnt += 1
        pydirectinput.press("down")
    pydirectinput.press("tab")


class FingerprintApp:
    def __init__(self, master, whole_templates=None, piece_templates=None):
        self.master = master
        master.title("GTAOL指纹破解助手")
        master.geometry("800x600")

        # 运行状态控制
        self.running = False
        self.thread = None
        self.templates_ready = False

        # 快捷键录入状态: None / "start" / "end"
        self.capturing = None
        self._capture_backup = ""

        # 先读配置，再用它初始化界面。
        # _settings 是给后台线程读的普通字典快照，避免工作线程直接碰 Tk 控件。
        self.config, config_warning = load_config()
        self._settings = {
            "start_key": self.config["start_key"],
            "end_key": self.config["end_key"],
            "interval": self.config["interval"],
        }

        # 创建GUI组件
        self.create_widgets()
        self.setup_bindings()

        if config_warning:
            self.log_message(f"配置: {config_warning}")
        else:
            self.log_message(f"配置已载入: {os.path.basename(CONFIG_PATH)}")

        # 启动时预加载所有模板到内存（只读一次磁盘）
        self.load_templates(whole_templates, piece_templates)

    def create_widgets(self):
        # 配置区域
        config_frame = ttk.LabelFrame(self.master, text="配置设置")
        config_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(config_frame, text="开始快捷键:").grid(row=0, column=0, padx=5, pady=(6, 0))
        self.start_key_var = tk.StringVar(value=self.config["start_key"])
        self.start_key_entry = self._make_hotkey_entry(config_frame, "start")
        self.start_key_entry.grid(row=0, column=1, padx=5, pady=(6, 0))

        ttk.Label(config_frame, text="结束快捷键:").grid(row=0, column=2, padx=5, pady=(6, 0))
        self.end_key_var = tk.StringVar(value=self.config["end_key"])
        self.end_key_entry = self._make_hotkey_entry(config_frame, "end")
        self.end_key_entry.grid(row=0, column=3, padx=5, pady=(6, 0))

        ttk.Label(config_frame, text="模拟操作间隔(s):").grid(row=0, column=4, padx=5, pady=(6, 0))
        self.interval = ttk.Entry(config_frame, width=8, justify="center")
        self.interval.insert(0, f"{self.config['interval']:g}")
        self.interval.grid(row=0, column=5, padx=5, pady=(6, 0))
        # 间隔仍是普通文本框，失去焦点或回车时保存
        self.interval.bind("<FocusOut>", lambda _event: self.save_current_config())
        self.interval.bind("<Return>", lambda _event: self.save_current_config())

        # 状态显示
        self.status = ttk.Label(self.master, text="当前状态: 等待开始", foreground="gray")
        self.status.pack(pady=5)

        # 日志区域
        log_frame = ttk.LabelFrame(self.master, text="运行日志")
        log_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.log = tk.Text(log_frame, height=10, state="disabled")
        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=vsb.set)

        self.log.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 控制按钮
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="开始运行", command=self.toggle_run)
        self.start_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="退出程序", command=self.cleanup).pack(side="left")

    def setup_bindings(self):
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup)
        # 快捷键录入统一在顶层窗口监听按键，免得只读输入框把按键事件吞掉
        self.master.bind("<KeyPress>", self.on_key_press, add="+")

    # ---------------- 快捷键录入 ----------------

    def _make_hotkey_entry(self, master, which):
        """生成一个只读的快捷键输入框。

        点它 -> 进入录入状态 -> 按下想用的键即可，不用手动打字。
        """
        var = self.start_key_var if which == "start" else self.end_key_var
        entry = ttk.Entry(master, textvariable=var, width=12,
                          state="readonly", justify="center", cursor="hand2")
        entry.bind("<Button-1>", lambda _event, w=which: self.begin_capture(w))
        entry.bind("<FocusIn>", lambda _event, w=which: self.begin_capture(w))
        entry.bind("<FocusOut>", lambda _event, w=which: self.cancel_capture(w))
        return entry

    def _hotkey_var(self, which):
        return self.start_key_var if which == "start" else self.end_key_var

    def begin_capture(self, which):
        """进入录入状态，等待用户按键。"""
        if self.capturing == which:
            return
        var = self._hotkey_var(which)
        self.capturing = which
        self._capture_backup = var.get()
        var.set("请按键…")
        self.status.config(text="当前状态: 请按下想要的按键（Esc 取消）", foreground="blue")

    def cancel_capture(self, which):
        """焦点离开但还没按键 -> 恢复原值，等于取消。"""
        if self.capturing != which:
            return
        self._hotkey_var(which).set(self._capture_backup)
        self.capturing = None
        self._restore_status()
        self.log_message("已取消快捷键修改")

    def _restore_status(self):
        if self.running:
            self.status.config(text="当前状态: 运行中", foreground="green")
        else:
            self.status.config(text="当前状态: 等待开始", foreground="gray")

    def on_key_press(self, event):
        """全局按键回调：只有正在录入快捷键时才处理。"""
        if not self.capturing:
            return

        which = self.capturing
        var = self._hotkey_var(which)

        # Esc 取消
        if event.keysym == "Escape":
            var.set(self._capture_backup)
            self.capturing = None
            self._restore_status()
            self.log_message("已取消快捷键修改")
            return "break"

        key = normalize_keysym(event.keysym)
        if key is None:
            return "break"      # 不认识的键，忽略并继续等待

        other = self._hotkey_var("end" if which == "start" else "start").get()
        if key == other:
            self.log_message(f"快捷键冲突: {key} 已经用在另一端，请换一个")
            var.set(self._capture_backup)
            self.capturing = None
            self._restore_status()
            return "break"

        var.set(key)
        self.capturing = None
        self._restore_status()
        self.save_current_config()
        self.log_message(f"快捷键已改为 {key} 并保存到 {os.path.basename(CONFIG_PATH)}")
        return "break"

    # ---------------- 配置保存 ----------------

    def _snapshot_settings(self):
        """把配置同步成普通字典，供后台线程安全读取。"""
        self._settings = {
            "start_key": self.config["start_key"],
            "end_key": self.config["end_key"],
            "interval": self.config["interval"],
        }

    def save_current_config(self):
        """界面当前设置 -> self.config -> config.json。"""
        # 正在录入快捷键时，输入框里是"请按键…"占位符，要用备份值
        start = self._capture_backup if self.capturing == "start" else self.start_key_var.get()
        end = self._capture_backup if self.capturing == "end" else self.end_key_var.get()
        self.config["start_key"] = start.lower()
        self.config["end_key"] = end.lower()

        interval_ok = True
        try:
            interval = float(self.interval.get())
            if interval < 0:
                raise ValueError
            self.config["interval"] = interval
        except (TypeError, ValueError):
            interval_ok = False
            self.log_message(f"操作间隔 “{self.interval.get()}” 不是有效数字，本次不保存该项")
        except Exception:
            # 窗口已经销毁等情况，静默跳过（退出时可能触发）
            interval_ok = False

        self._snapshot_settings()

        if not interval_ok:
            return
        try:
            save_config(self.config)
        except Exception as exc:
            self.log_message(f"配置文件保存失败: {exc}")

    def log_message(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def load_templates(self, whole_templates=None, piece_templates=None):
        """启动时预加载模板；失败就在日志里报到具体原因，并禁用开始按钮。"""
        global WHOLE_TEMPLATES, PIECE_TEMPLATES
        try:
            if whole_templates is None or piece_templates is None:
                whole_templates, piece_templates = load_all_templates()
            WHOLE_TEMPLATES = whole_templates
            PIECE_TEMPLATES = piece_templates
            self.templates_ready = True
            total = len(WHOLE_TEMPLATES) + sum(len(g) for g in PIECE_TEMPLATES.values())
            self.log_message(f"模板预加载完成: 共 {total} 个（4 全图 + 16 碎片），已全部驻留内存")
            self.log_message(
                f"阈值: 全图 {WHOLE_THRESHOLD:.2f} / 碎片 {PIECE_THRESHOLD:.2f}；"
                f"滑动余量 {WHOLE_SEARCH_MARGIN}px / {PIECE_SEARCH_MARGIN}px"
            )
        except Exception as e:
            self.templates_ready = False
            self.log_message(f"模板预加载失败: {e}")
            self.status.config(text="当前状态: 模板缺失，无法运行", foreground="red")
            self.start_btn.config(state="disabled")

    def toggle_run(self):
        if not self.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        if not self.templates_ready:
            self.log_message("模板未就绪，无法开始运行")
            return
        # 用界面上的最新设置并落盘，再启动工作线程
        self.save_current_config()
        self.running = True
        self.status.config(text="当前状态: 运行中", foreground="green")
        self.start_btn.config(text="停止运行")
        self.thread = threading.Thread(target=self.main_loop, daemon=True)
        self.thread.start()

    def stop_processing(self):
        self.running = False
        self.status.config(text="当前状态: 已停止", foreground="red")
        self.start_btn.config(text="开始运行")

    def main_loop(self):
        while self.running:
            # 正在录入快捷键时先不响应，免得改键按下的那个键把流程触发了
            if self.capturing:
                time.sleep(0.1)
                continue

            # 每轮都从快照里取最新设置：改完快捷键立即生效，不用重启程序。
            # 这里只读普通字典，不直接访问 Tk 控件。
            settings = self._settings
            pydirectinput.PAUSE = settings["interval"]

            if keyboard.is_pressed(settings["start_key"]):
                try:
                    self.process_fingerprint()
                except Exception as e:
                    self.log_message(f"错误发生: {str(e)}")
            elif keyboard.is_pressed(settings["end_key"]):
                self.stop_processing()
            time.sleep(0.1)

    def process_fingerprint(self):
        """完整识别流程：只截一次图 -> 裁全图 -> 匹配 -> 裁 8 碎片 -> 逐个匹配 -> control()"""
        start_time = time.time()
        global marked
        self.log_message("开始处理指纹...")

        try:
            # 1) 给游戏留出绘制指纹界面的时间，然后**只截一次**整个游戏窗口
            time.sleep(UI_SETTLE_DELAY)
            img_base = get_game_window_img()

            info = get_last_window_info()
            if info.get("aspect_warning"):
                self.log_message(f"警告: {info['aspect_warning']}")

            # 2) 从同一张基准图里裁剪指纹大图，找当前是哪一组
            whole_gray = cv2.cvtColor(get_whole_fingerprint(img_base), cv2.COLOR_BGR2GRAY)
            whole_debug = {}
            index = Match_whole_pic(whole_gray, debug=whole_debug)
            self.log_message(
                "全图分数: " + " ".join(f"[{i}]{s:.3f}" for i, s in whole_debug["scores"])
            )

            if index == -1:
                self.log_message(
                    f"未找到匹配的指纹！（最高分 {whole_debug['best_score']:.3f} "
                    f"< 阈值 {WHOLE_THRESHOLD:.2f}）"
                )
                winsound.MessageBeep()
                return

            # 3) 8 个碎片同样从这一张基准图里裁剪，不再重复截图、不再重复读盘
            piece_imgs = get_fingerprint_pics(img_base)

            marked = [False] * 4
            fingerprints = [False] * 8

            for i, piece_bgr in enumerate(piece_imgs):
                piece_gray = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)
                piece_debug = {}
                if Match_piece_pic(piece_gray, index, marked, debug=piece_debug):
                    fingerprints[i] = True
                    self.log_message(
                        f"  槽位{i} 命中碎片{piece_debug['best_index']} "
                        f"(分数 {piece_debug['best_score']:.3f})"
                    )
                else:
                    score = piece_debug.get("best_score", -1.0)
                    self.log_message(f"  槽位{i} 不匹配 (最高分 {score:.3f})")

            control(fingerprints)

            cost = time.time() - start_time
            self.log_message(f"处理完成，耗时: {cost:.2f}秒")

        except Exception as e:
            self.log_message(f"处理过程中发生错误: {str(e)}")

    def cleanup(self):
        if getattr(self, "_closing", False):   # 防止"退出按钮 + 关窗口"重复触发
            return
        self._closing = True
        self.stop_processing()
        self.save_current_config()   # 退出前把当前设置（含间隔）落盘
        release_resources()
        self.master.destroy()


def main():
    """程序入口：先建界面（界面内部会预加载模板），再进入事件循环。"""
    root = tk.Tk()
    FingerprintApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
