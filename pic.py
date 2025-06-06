import time
from PIL import ImageGrab

# 分辨率2560*1440, 全屏
# 列优先，上到下左侧0-4右侧5-7
TOP_LEFT = [(640, 370), (640, 561), (640, 753), (640, 946),
            (833, 370), (833, 561), (833, 753), (833, 946)]   # 各块左上起始区域
BOTTOM_RIGHT = [(781, 511), (781, 702), (781, 894), (781, 1087),
                (974, 511), (974, 702), (974, 894), (974, 1087)]   # 各块右下终止区域

class GetPic():
    def __init__(self):
        self.pic = None

    def get_area_shot(top_left, bottom_right):   #获取区域截图
        area = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        screenshot = ImageGrab.grab(area)
        # screenshot.show()
        return screenshot

def get_fingerprint_pics():
    """
    获取指纹块大图集
    """
    height_space = 191
    width_space = 193
    pics = []
    for i in range(2):
        for j in range(4):
            top_left, bottom_right = TOP_LEFT[i * 4 + j], BOTTOM_RIGHT[i * 4 + j]
            pic = GetPic.get_area_shot(top_left, bottom_right)
            pics.append(pic)
            # pic.save(f"pic/fingerprint_{start_finger * 8 + i * 4 + j}.png") # 存储各图
    return pics

def get_whole_fingerprint():
    """
    获取指纹大图
    """
    time.sleep(3)
    START_POINT = (1300, 205)
    END_POINT = (1760, 880)
    pic = GetPic.get_area_shot(START_POINT, END_POINT)
    # pic.save(f"pic/fingerprint{start_finger}.png")
    return pic

if __name__ == "__main__":
    start_finger = 0
    while True:
        time.sleep(3)
        get_fingerprint_pics()
        # get_whole_fingerprint()
        input()
        start_finger += 1    # 手动调整指纹数