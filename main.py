import cv2
import time
import pydirectinput
import os
import keyboard 
import winsound
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

from pic import get_whole_fingerprint, get_fingerprint_pics

'''
preloaded_images = {}
for idx in range(4):
    for piece in range(4):
        path = f"pic/fingerprint{idx}/fingerprint{idx}_{piece}.png"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        preloaded_images[(idx, piece)] = img

marked_lock = threading.Lock()
'''

class FingerprintApp:
    def __init__(self, master):
        self.master = master
        master.title("GTAOL指纹破解助手")
        master.geometry("600x400")

        # 运行状态控制
        self.running = False
        self.thread = None
        
        # 创建GUI组件
        self.create_widgets()
        self.setup_bindings()

    def create_widgets(self):
        # 配置区域
        config_frame = ttk.LabelFrame(self.master, text="配置设置")
        config_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(config_frame, text="开始快捷键:").grid(row=0, column=0, padx=5)
        self.start_key = ttk.Entry(config_frame, width=10)
        self.start_key.insert(0, "f11")
        self.start_key.grid(row=0, column=1)

        ttk.Label(config_frame, text="结束快捷键:").grid(row=0, column=2, padx=5)
        self.end_key = ttk.Entry(config_frame, width=10)
        self.end_key.insert(0, "f12")
        self.end_key.grid(row=0, column=3)

        ttk.Label(config_frame, text="模拟操作间隔(s):").grid(row=0, column=4, padx=5)
        self.interval = ttk.Entry(config_frame, width=8)
        self.interval.insert(0, "0.02")
        self.interval.grid(row=0, column=5)

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

    def log_message(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def toggle_run(self):
        if not self.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
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
        pydirectinput.PAUSE = float(self.interval.get())
        start_key = self.start_key.get().lower()
        end_key = self.end_key.get().lower()

        while self.running:
            if keyboard.is_pressed(start_key):
                try:
                    self.process_fingerprint()
                except Exception as e:
                    self.log_message(f"错误发生: {str(e)}")
            elif keyboard.is_pressed(end_key):
                self.stop_processing()
            time.sleep(0.1)

    def process_fingerprint(self):
        start_time = time.time()
        # winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        self.log_message("开始处理指纹...")

        try:
            fingerprint = np.array(get_whole_fingerprint())
            fingerprint = cv2.cvtColor(fingerprint, cv2.COLOR_RGB2GRAY)
            # print(fingerprint)
            index = Match_whole_pic(fingerprint)
            
            if index == -1:
                self.log_message("未找到匹配的指纹！")
                winsound.MessageBeep()
                return

            pics = get_fingerprint_pics()
            for i in range(len(pics)):
                pics[i] = cv2.cvtColor(np.array(pics[i]), cv2.COLOR_RGB2GRAY)
            
            global marked
            marked = [False] * 4
            fingerprints = [False] * 8
            
            for i in range(len(pics)):
                if Match_piece_pic(pics[i], index):
                    fingerprints[i] = True

            control(fingerprints)
            
            cost = time.time() - start_time
            self.log_message(f"处理完成，耗时: {cost:.2f}秒")

        except Exception as e:
            self.log_message(f"处理过程中发生错误: {str(e)}")

    def cleanup(self):
        self.stop_processing()
        self.master.destroy()

# 使用SSIM进行匹配

'''
def Match_piece_pic(pic, index):
    """
    多线程对比碎片图判断是否一致
    """

    image1 = pic
    candidates = []
    
    # 获取候选索引（线程安全）
    with marked_lock:
        candidates = [i for i in range(4) if not marked[i]]
    
    if not candidates:
        return False
    
    def process_candidate(i):
        """并行处理单个候选碎片"""
        image2 = preloaded_images.get((index, i))
        if image2 is None:
            return (i, -1)  # 处理文件不存在的情况
        score, _ = ssim(image1, image2, full=True)
        return (i, score)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 为每个候选创建future
        future_to_i = {executor.submit(process_candidate, i): i for i in candidates}
        
        try:
            for future in as_completed(future_to_i):
                i, score = future.result()
                if score >= 0.85:
                    with marked_lock:
                        # 双重检查防止竞争条件
                        if not marked[i]:
                            marked[i] = True
                            # 取消剩余任务
                            for f in future_to_i:
                                f.cancel()
                            return True
        except Exception as e:
            print(f"Error during processing: {str(e)}")
    return False
'''

def Match_whole_pic(fingerprint):
    """
    对比两张大图找到所在指纹集
    """
    image1 = fingerprint
    for i in range(4):
        image2 = cv2.imread(f"pic/fingerprint{i}/fingerprint{i}.png", cv2.IMREAD_GRAYSCALE)
        score, _ = ssim(image1, image2, full=True)
        # print(f"ssim score: {score}")
        if score >= 0.95:
            return i
    else:
        return -1

def Match_piece_pic(pic, index):
    """
    对比碎片图判断是否一致
    """
    image1 = pic
    for i in range(4):
        if not marked[i]:
            image2 = cv2.imread(f"pic/fingerprint{index}/fingerprint{index}_{i}.png", cv2.IMREAD_GRAYSCALE)
            score, _ = ssim(image1, image2, full=True)
            if score >= 0.85:   # 0.90由于最底下背景闪动可能出现无法判断
                marked[i] = True
                return True
            # print(f"{index}_{i} score: {score}")
    else:
        return False

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

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()