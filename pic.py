# -*- coding: utf-8 -*-
"""
pic.py —— 截图与裁剪模块（重构版）

设计要点
--------
1. 不再假设"游戏必须全屏跑在 2560x1440 且窗口不能动"：
   先用 pygetwindow 找到 GTA V 窗口，再用 mss 只截这个窗口区域，
   最后等比缩放到基准分辨率 2560x1440。之后的裁剪坐标
   （TOP_LEFT / BOTTOM_RIGHT / START_POINT / END_POINT）保持原值不变，
   因为它们本来就是相对"基准分辨率"量出来的。

2. 一次识别只截一次图：8 个碎片全部从同一张内存图里裁剪，
   不再像旧代码那样调用 9 次 ImageGrab（9 次独立的屏幕抓取）。

3. get_whole_fingerprint(img_base) / get_fingerprint_pics(img_base) 是新的主接口，
   从基准图里按原坐标裁剪；旧接口保留为 *_legacy()，行为与旧版完全一致
   （PIL 全屏绝对坐标截图），方便对照排查。

坐标系约定：所有常量都是 2560x1440 基准分辨率下的像素坐标。
"""

import ctypes
import threading
import time

import cv2
import numpy as np

try:
    import mss
except ImportError as exc:  # 依赖缺失时给出明确指引，而不是抛裸 ImportError
    raise ImportError("缺少依赖 mss，请先执行: pip install mss") from exc

try:
    import pygetwindow as gw
except ImportError as exc:
    raise ImportError("缺少依赖 pygetwindow，请先执行: pip install pygetwindow") from exc

from PIL import ImageGrab  # 仅供 *_legacy() 使用


# ==========================================================================
# 基准坐标系（模板就是按这个分辨率采集的，以下数值一律不要改）
# ==========================================================================

BASE_WIDTH, BASE_HEIGHT = 2560, 1440

# 列优先，上到下左侧 0-4 右侧 5-7
TOP_LEFT = [(640, 370), (640, 561), (640, 753), (640, 946),
            (833, 370), (833, 561), (833, 753), (833, 946)]   # 各块左上起始区域
BOTTOM_RIGHT = [(781, 511), (781, 702), (781, 894), (781, 1087),
                (974, 511), (974, 702), (974, 894), (974, 1087)]   # 各块右下终止区域

START_POINT = (1300, 205)   # 指纹大图左上
END_POINT = (1760, 880)     # 指纹大图右下

# 单个碎片槽位的标准尺寸（由上面的常量推出：141x141）
PIECE_WIDTH = BOTTOM_RIGHT[0][0] - TOP_LEFT[0][0]
PIECE_HEIGHT = BOTTOM_RIGHT[0][1] - TOP_LEFT[0][1]
# 指纹大图的标准尺寸（460x675）
WHOLE_WIDTH = END_POINT[0] - START_POINT[0]
WHOLE_HEIGHT = END_POINT[1] - START_POINT[1]

TARGET_ASPECT = BASE_WIDTH / BASE_HEIGHT   # 16:9 ≈ 1.7778
ASPECT_TOLERANCE = 0.03                    # 宽高比偏差超过 3% 时在日志里告警

WINDOW_TITLE_KEYWORD = "Grand Theft Auto V"
MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT = 640, 360   # 小于这个尺寸基本不可能是游戏主窗口

SCREENSHOT_DELAY = 3.0        # 仅旧的 *_legacy() 接口使用（等价旧版 sleep(3)）
ENABLE_DPI_AWARENESS = True   # 见下方 _enable_dpi_awareness() 的说明


# ==========================================================================
# 一、DPI 感知
# ==========================================================================

def _enable_dpi_awareness():
    """把当前进程标记为"DPI 感知"，必须在任何截图/建窗之前调用。

    如果进程不是 DPI 感知的，而系统缩放又不是 100%，Windows 会对进程做
    DPI 虚拟化：GetWindowRect 返回缩放后的逻辑坐标，屏幕位图也会被整体
    缩放。两者虽然自洽（裁剪位置仍然对），但截图会明显发虚，相关性得分
    会掉，这正是我们要避免的准确率损失。

    注意：开启后 Tk 界面在高缩放比下会显得更小但更清晰。
    若希望 GUI 保持系统缩放的外观，把 ENABLE_DPI_AWARENESS 改成 False
    （代价是缩放比不是 100% 时识别得分会下降）。
    """
    if not ENABLE_DPI_AWARENESS:
        return False
    try:
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return True
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return True
    except Exception:
        return False


DPI_AWARE = _enable_dpi_awareness()


# ==========================================================================
# 二、窗口定位（带缓存）
# ==========================================================================

_window_lock = threading.Lock()
_cached_window = None          # 缓存窗口对象本身，避免每次识别都枚举全部窗口
_last_window_info = {}         # 最近一次截图的窗口信息，供上层写日志


def _read_window_metrics(win):
    """读取窗口矩形。

    pygetwindow 的 left/top/width/height 每次读取都会重新调用 Win32
    GetWindowRect（属性背后挂了 rect 的 onRead 回调），所以缓存窗口对象
    是安全的：窗口被拖动后，下一次读取拿到的仍是最新位置。
    """
    left, top = int(win.left), int(win.top)
    width, height = int(win.width), int(win.height)
    return left, top, width, height


def _window_is_usable(win):
    """窗口是否可用作截图目标：未最小化且尺寸正常。"""
    try:
        if win.isMinimized:
            return False
        _, _, width, height = _read_window_metrics(win)
    except Exception:
        # 窗口刚被关闭时 GetWindowRect 会失败，直接视为不可用
        return False
    return width >= MIN_WINDOW_WIDTH and height >= MIN_WINDOW_HEIGHT


def find_game_window(force_refresh=False):
    """查找 GTA V 游戏窗口，返回 pygetwindow 的窗口对象。

    结果会被缓存；只有当缓存失效（窗口关闭/最小化/尺寸异常）或显式要求
    force_refresh 时才重新枚举窗口列表。

    找不到时抛 RuntimeError，并说明可能的原因。
    """
    global _cached_window

    with _window_lock:
        if _cached_window is not None and not force_refresh:
            if _window_is_usable(_cached_window):
                return _cached_window
            _cached_window = None

        try:
            candidates = gw.getWindowsWithTitle(WINDOW_TITLE_KEYWORD)
        except Exception as exc:
            raise RuntimeError(f"枚举窗口失败: {exc}") from exc

        usable = [w for w in candidates if _window_is_usable(w)]
        if not usable:
            raise RuntimeError(
                f"未找到标题包含 “{WINDOW_TITLE_KEYWORD}” 的可用窗口。\n"
                f"  候选窗口数: {len(candidates)}（可能全部被最小化或尺寸过小）\n"
                f"  请确认：1) GTA V 已启动并进入游戏画面；"
                f"2) 使用「窗口化」或「无边框窗口」模式——独占全屏下系统截图通常是黑屏。"
            )

        # 同名窗口可能不止一个（例如启动器），取面积最大的那个
        _cached_window = max(usable, key=lambda w: w.width * w.height)
        return _cached_window


def get_window_info(win=None):
    """返回窗口信息字典（不截图），用于日志和诊断。"""
    win = win or find_game_window()
    left, top, width, height = _read_window_metrics(win)
    if width < MIN_WINDOW_WIDTH or height < MIN_WINDOW_HEIGHT:
        raise RuntimeError(
            f"游戏窗口尺寸异常: {width}x{height}，"
            f"至少需要 {MIN_WINDOW_WIDTH}x{MIN_WINDOW_HEIGHT}。"
            f"请把 GTA V 切到前台并恢复正常窗口/无边框窗口模式。"
        )
    aspect = width / float(height)
    deviation = abs(aspect - TARGET_ASPECT) / TARGET_ASPECT
    info = {
        "title": win.title,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "scale_x": BASE_WIDTH / float(width),
        "scale_y": BASE_HEIGHT / float(height),
        "aspect": aspect,
        "aspect_deviation": deviation,
        "aspect_warning": None,
    }
    if deviation > ASPECT_TOLERANCE:
        info["aspect_warning"] = (
            f"游戏窗口宽高比 {aspect:.4f} 偏离 16:9（{TARGET_ASPECT:.4f}）达 "
            f"{deviation * 100:.1f}%，画面被拉伸会让裁剪位置偏移；"
            f"建议把 GTA V 设为 16:9 分辨率（如 2560x1440 / 1920x1080）。"
        )
    return info


# ==========================================================================
# 三、截图与缩放
# ==========================================================================

_tls = threading.local()   # mss 实例不是线程安全的，按线程各留一份


def _get_sct():
    """取得当前线程的 mss 实例（懒加载）。"""
    sct = getattr(_tls, "sct", None)
    if sct is None:
        sct = mss.mss()
        _tls.sct = sct
    return sct


def release_resources():
    """释放当前线程持有的截图资源（退出程序时调用）。"""
    sct = getattr(_tls, "sct", None)
    if sct is not None:
        try:
            sct.close()
        except Exception:
            pass
        _tls.sct = None


def _resize_to_base(bgr):
    """把窗口截图等比缩放到基准分辨率 2560x1440。"""
    height, width = bgr.shape[:2]
    if width == BASE_WIDTH and height == BASE_HEIGHT:
        return bgr

    # 缩小方向用 INTER_AREA（正确的降采样滤波器）；
    # 放大方向不能用 INTER_AREA —— OpenCV 在放大时它会退化成最近邻，
    # 实测 1080p 窗口下相关性 0.9897 vs INTER_CUBIC 的 0.9944，
    # 720p 下 0.9724 vs 0.9828，差别足以影响阈值余量。
    if width >= BASE_WIDTH and height >= BASE_HEIGHT:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(bgr, (BASE_WIDTH, BASE_HEIGHT), interpolation=interpolation)


def get_game_window_img():
    """截取游戏窗口并返回**基准分辨率下的 BGR 图**（numpy 数组）。

    失败时抛 RuntimeError，消息里说明具体原因（窗口未找到 / 尺寸异常 /
    截图失败）。
    """
    global _last_window_info

    win = find_game_window()
    info = get_window_info(win)

    sct = _get_sct()
    rect = {"left": info["left"], "top": info["top"],
            "width": info["width"], "height": info["height"]}

    try:
        shot = sct.grab(rect)
    except Exception as exc:
        # 窗口可能在两次调用之间被移动/关闭，刷新缓存后重试一次
        win = find_game_window(force_refresh=True)
        info = get_window_info(win)
        rect = {"left": info["left"], "top": info["top"],
                "width": info["width"], "height": info["height"]}
        try:
            shot = sct.grab(rect)
        except Exception as exc2:
            raise RuntimeError(
                f"截取游戏窗口失败: {exc2}（窗口区域 {rect}）。"
                f"如果是独占全屏，请改成「无边框窗口」模式。"
            ) from exc2

    frame = np.asarray(shot)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(f"截图数据维度异常: {frame.shape}，期望 (H, W, 3) 或 (H, W, 4)")

    # mss 返回 BGRA，丢掉 alpha 得到 BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame[:, :, :3]

    _last_window_info = dict(info)

    if bgr.shape[1] != BASE_WIDTH or bgr.shape[0] != BASE_HEIGHT:
        bgr = _resize_to_base(bgr)

    return bgr


def get_last_window_info():
    """最近一次成功截图时的窗口信息（含 aspect_warning），供上层写日志。"""
    return dict(_last_window_info)


# ==========================================================================
# 四、按基准坐标裁剪（带边界保护）
# ==========================================================================

def _crop_base(img_base, top_left, bottom_right, label):
    """从基准图里裁剪一块区域，越界时抛出明确的错误。"""
    if img_base is None:
        raise RuntimeError(f"{label}: 传入的基准图为 None")
    if img_base.ndim not in (2, 3):
        raise RuntimeError(f"{label}: 基准图维度异常 {img_base.shape}")

    height, width = img_base.shape[:2]
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RuntimeError(
            f"{label} 裁剪区域越界: ({x1},{y1})-({x2},{y2}) 超出基准图 {width}x{height}。"
            f"基准图应为 {BASE_WIDTH}x{BASE_HEIGHT}，请检查截图缩放是否正常。"
        )

    crop = img_base[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"{label} 裁剪结果为空: ({x1},{y1})-({x2},{y2})")
    return crop


def get_whole_fingerprint(img_base):
    """从基准图中裁剪指纹大图（BGR）。

    img_base: get_game_window_img() 返回的基准分辨率图像。
    """
    return _crop_base(img_base, START_POINT, END_POINT, "指纹大图")


def get_fingerprint_pics(img_base):
    """从基准图中裁剪 8 个碎片槽位，返回长度 8 的 BGR 图列表。

    顺序与 TOP_LEFT / BOTTOM_RIGHT 一致（列优先，左 0-3 右 4-7）。
    """
    pics = []
    for i in range(8):
        pics.append(_crop_base(img_base, TOP_LEFT[i], BOTTOM_RIGHT[i], f"碎片槽位{i}"))
    return pics


# ==========================================================================
# 五、旧接口（行为与重构前完全一致，仅用于兼容与对照排查）
# ==========================================================================

class GetPic():
    """与旧版一致的截图小工具，保留下来给 *_legacy() 使用。"""

    def __init__(self):
        self.pic = None

    def get_area_shot(top_left, bottom_right):   # 获取区域截图
        area = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        screenshot = ImageGrab.grab(area)
        # screenshot.show()
        return screenshot


def get_whole_fingerprint_legacy():
    """旧版：全屏绝对坐标 + PIL 截图，返回 PIL.Image（RGB）。"""
    time.sleep(SCREENSHOT_DELAY)
    return GetPic.get_area_shot(START_POINT, END_POINT)


def get_fingerprint_pics_legacy():
    """旧版：全屏绝对坐标 + 8 次 PIL 截图，返回 PIL.Image 列表。"""
    pics = []
    for i in range(2):
        for j in range(4):
            top_left, bottom_right = TOP_LEFT[i * 4 + j], BOTTOM_RIGHT[i * 4 + j]
            pics.append(GetPic.get_area_shot(top_left, bottom_right))
    return pics


if __name__ == "__main__":
    # 手动调试用：打印窗口信息并保存一张基准图
    info = get_window_info()
    print("窗口信息:", info)
    base = get_game_window_img()
    print("基准图尺寸:", base.shape, " BGR")
    cv2.imwrite("_debug_base.png", base)
    cv2.imwrite("_debug_whole.png", get_whole_fingerprint(base))
    for n, p in enumerate(get_fingerprint_pics(base)):
        cv2.imwrite(f"_debug_piece{n}.png", p)
    print("已保存 _debug_base.png / _debug_whole.png / _debug_piece*.png")
