import os
import cv2
import glob
from utils.logger import logger

def resize_if_needed(frame):
    """
    检查图片尺寸，如果超过1920x1080则按比例缩放，否则保持原图
    """
    height, width = frame.shape[:2]
    
    # 如果图片尺寸在允许范围内，直接返回原图
    if width <= 1920 and height <= 1080:
        return frame
        
    # 计算缩放比例，保持宽高比
    scale = min(1920/width, 1080/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 使用高质量的插值算法进行缩放
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    logger.info(f"图片尺寸 {width}x{height} 超过限制，已缩放至 {new_width}x{new_height}")
    return resized_frame

class ImageStream:
    def __init__(self, source=0):
        self.path = source

    def read(self):
        if not os.path.exists(self.path):
            return None

        basename = os.path.basename(self.path).lower()
        if not basename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return None

        frame = cv2.imread(self.path)
        if frame is None:
            return None
            
        # 检查图片尺寸，如果超过限制则按比例缩放
        return resize_if_needed(frame)

class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.is_video = True

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
            
        # 检查视频帧尺寸，如果超过限制则按比例缩放
        return resize_if_needed(frame)

    def release(self):
        self.cap.release()

class FolderStream:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = []
        self.current_index = 0
        self.is_video = False
        
        # 获取文件夹中所有支持的图片文件
        supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for format in supported_formats:
            self.image_files.extend(glob.glob(os.path.join(folder_path, format)))
            self.image_files.extend(glob.glob(os.path.join(folder_path, format.upper())))
        
        # 按文件名排序
        self.image_files.sort()
        logger.info(f"文件夹中找到 {len(self.image_files)} 张图片")

    def read(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            return None
            
        image_path = self.image_files[self.current_index]
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # 检查图片尺寸，如果超过限制则按比例缩放
            frame = resize_if_needed(frame)
            logger.info(f"读取图片: {os.path.basename(image_path)} ({self.current_index + 1}/{len(self.image_files)})")
        self.next()
        return frame
    
    def next(self):
        """移动到下一张图片"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        else: 
            self.reset()
            return True
    
    def reset(self):
        """重置到第一张图片"""
        self.current_index = 0