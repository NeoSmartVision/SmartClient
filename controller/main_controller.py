
from multiprocessing import Value
from tkinter import NO
import os
import cv2
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QLinearGradient
from model.stream import ImageStream, VideoStream, FolderStream
from model.inference import Inference
from model.draw import ImageDraw
from PyQt5.QtGui import QImage, QPixmap
from utils.logger import logger
import numpy as np
import threading
from model.server import start_server, stop_server, stop_all_servers

class MainController:
    def __init__(self, ui):
        self.ui = ui
        self.model = None
        self.stream = None
        self.file_name = None
        self.roi_points = []
        self.drawer = ImageDraw()
        self.current_server = None

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.process_current_frame)

        
        self.mode = 'image' # choice = ['image', 'folder', 'video']
        self.api = {
            '03_vehicle': 'http://127.0.0.1:5000/sv/tracking_vehicle',      
            '06_person': 'http://127.0.0.1:5000/sv/tracking_person',
            '07_fire': 'http://127.0.0.1:5000/sv/detection_fire',
            '08_smoke': 'http://127.0.0.1:5000/sv/detection_smoke',
            '09_roadmanhole': 'http://127.0.0.1:5000/sv/detection_roadmanhole',
            '10_roadpothole': 'http://127.0.0.1:5000/sv/detection_roadpothole',
            '11_roadcrack': 'http://127.0.0.1:5000/sv/detection_roadcrack',
            '12_roadwater': 'http://127.0.0.1:5000/sv/detection_roadwaterlogging',
            '13_animal': 'http://127.0.0.1:5000/sv/detection_animal',
            '14_overflow': 'http://127.0.0.1:5000/sv/detection_garbageoverflow',
            '15_roadcongestion': 'http://127.0.0.1:5000/sv/detection_roadcongestion',
            '16_illegalparking': 'http://127.0.0.1:5000/sv/tracking_illegalparking',
            '17_licenseplate': 'http://127.0.0.1:5000/sv/recognition_licenseplate',
            '18_slagtruck': 'http://127.0.0.1:5000/sv/detection_slagtruck',
            '19_elevator': 'http://127.0.0.1:5000/sv/detection_elevator',
            '20_employeeabsence': 'http://127.0.0.1:5000/sv/detection_employeeabsence',
            '21_nvmencroachment': 'http://127.0.0.1:5000/sv/detection_nmvencroachment',
            '22_face': 'http://127.0.0.1:5000/sv/recognition_face',


            '53_droneroadcrack':'http://127.0.0.1:5000/sv/detection_droneroadcrack',
            '56_roadstall':'http://127.0.0.1:5000/sv/detection_roadstall',
            '57_riverfloatingdebris':'http://127.0.0.1:5000/sv/detection_riverfloatingdebris',
        }

        # 追加页面信息
        self.ui.selectModelBox.addItems(self.api.keys())
        self.ui.tableWidget.setColumnWidth(2, 150)
        row_count = self.ui.tableWidget.rowCount()
        for row in range(row_count):
            self.ui.tableWidget.setRowHeight(row, 50)  # 每行高度设置为 50

        # 绑定按钮事件
        self.ui.selectImageButton.clicked.connect(self.select_image)
        self.ui.selectVideoButton.clicked.connect(self.select_video)
        self.ui.selectFolderButton.clicked.connect(self.select_folder)
        self.ui.selectModelBox.currentIndexChanged.connect(self.select_model)
        self.ui.nmsSpinBox.valueChanged.connect(self.nmsspinbox_changed)
        self.ui.nmsSlider.valueChanged.connect(self.nmsslider_changed)
        self.ui.conSpinBox.valueChanged.connect(self.conspinbox_changed)
        self.ui.conSlider.valueChanged.connect(self.conslider_changed)

        # 稳定图像显示区域，避免设置像素图后改变sizeHint引发布局抖动
        if hasattr(self.ui, 'label') and isinstance(self.ui.label, QLabel):
            self.ui.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            # 设定一个合理的最小尺寸（可按需要调整）
            if self.ui.label.minimumWidth() == 0 and self.ui.label.minimumHeight() == 0:
                self.ui.label.setMinimumSize(640, 360)

        self.ui.startDetectionButton.clicked.connect(self.inference)
        self.ui.clearImageButton.clicked.connect(self.clear_image)
        self.ui.exportDataButton.clicked.connect(self.export_data)
        self.ui.setROIButton.clicked.connect(self.set_roi_image)
        self.ui.clearROIButton.clicked.connect(self.clear_roi_image)
        
    def select_image(self):
        self.file_name, _ = QFileDialog.getOpenFileName(
            self.ui, 
            "选择图片", 
            "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif)",
            options=QFileDialog.DontUseNativeDialog
        )
        if self.file_name:
            self.ui.label.setStyleSheet("background-color: white;")

            # 读取图片文件
            self.frame = cv2.imread(self.file_name)
            if self.frame is not None:
                self.display(self.frame)

        self.stream = ImageStream(self.file_name)
        self.mode = 'image'

    def select_video(self):
        self.file_name, _ = QFileDialog.getOpenFileName(
            self.ui, 
            "选择视频", 
            "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)",
            options=QFileDialog.DontUseNativeDialog
        )
        if self.file_name:
            self.ui.label.setStyleSheet("background-color: white;")
            
            # 创建视频流对象
            self.stream = VideoStream(self.file_name)
            
            # 读取第一帧并显示
            self.frame = self.stream.read()
            if self.frame is not None:
                self.display(self.frame)
                
                logger.info(f'视频文件已选择: {self.file_name}')
                self.mode = 'video'
            else:
                logger.error('无法读取视频文件')
        else:
            logger.warning('未选择视频文件')

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self.ui,
            "选择图片文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        )
        if folder_path:
            self.file_name = folder_path
            self.ui.label.setStyleSheet("background-color: white;")
            
            # 创建文件夹流对象
            self.stream = FolderStream(folder_path)
            
            # 读取第一张图片并显示
            self.frame = self.stream.read()
            if self.frame is not None:
                self.display(self.frame)
                
                logger.info(f'文件夹已选择: {folder_path}')
                self.mode = 'folder'
            else:
                logger.warning('文件夹中没有找到支持的图片文件')
        else:
            logger.warning('未选择文件夹')

    def select_model(self, index):
        stop_all_servers()

        model_name = self.ui.selectModelBox.itemText(index)
        self.thread = threading.Thread(
                target=start_server,
                args=(model_name,),
                daemon=True
            )
        self.thread.start()
        self.current_server = model_name

        self.model = Inference(self.api[model_name])
        logger.info(f"选择了模型: {model_name}")
    
    def nmsspinbox_changed(self, value):
        # 将 spinbox 的浮点值映射到 slider 的整数值
        self.ui.nmsSlider.blockSignals(True)
        self.ui.nmsSlider.setValue(int(value * 100))   # 根据实际范围调整
        self.ui.nmsSlider.blockSignals(False)
    
    def nmsslider_changed(self, value):
        # 将 slider 的整数值映射到 spinbox 的浮点值
        self.ui.nmsSpinBox.blockSignals(True)          # 阻止循环信号
        self.ui.nmsSpinBox.setValue(value / 100.0)     # 根据实际范围调整
        self.ui.nmsSpinBox.blockSignals(False)

    def conspinbox_changed(self, value):
        self.ui.conSlider.blockSignals(True)
        self.ui.conSlider.setValue(int(value * 100))   # 根据实际范围调整
        self.ui.conSlider.blockSignals(False)
    
    def conslider_changed(self, value):
        self.ui.conSpinBox.blockSignals(True)          # 阻止循环信号
        self.ui.conSpinBox.setValue(value / 100.0)     # 根据实际范围调整
        self.ui.conSpinBox.blockSignals(False)

    def inference(self):
        # 停止之前的视频定时器
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.ui.startDetectionButton.setText("开始检测")
            return

        if self.model is None:
            logger.warning("请先选择一个有效的模型")
            return
            
        if self.stream is None:
            logger.warning("请先选择图片、视频或文件夹")
            return

        if self.mode == 'video':
            # 视频模式：开始自动连续帧检测
            logger.info("开始视频连续检测...")
            self.video_timer.start(20)  # 每100ms处理一帧
            # 修改按钮文本为"停止检测"
            self.ui.startDetectionButton.setText("停止检测")
        elif self.mode == 'folder':
            logger.info("开始文件夹连续检测...")
            self.video_timer.start(2000)  # 每100ms处理一帧
            # 修改按钮文本为"停止检测"
            self.ui.startDetectionButton.setText("停止检测")
        else:
            # 图片或文件夹模式：处理当前帧
            self.process_current_frame()
    
    def process_current_frame(self):
        """处理当前帧（图片或文件夹中的图片）"""

        if self.mode not in ['folder', 'video']:
            self.video_timer.stop()

        self.frame = self.stream.read()
        if self.frame is None:
            if self.mode == 'video':
                # 视频结束或无法读取
                logger.error("视频检测完成或无法读取帧")
                self.video_timer.stop()
            else:
                logger.error("无法读取图像或视频帧")
            return
            
        try:
            result = self.model.run(self.frame,
                                    nms = self.ui.nmsSpinBox.value(),
                                    polygon = self.roi_points,
                                    confidence = self.ui.conSpinBox.value(),
                                    timeout = self.ui.timeoutSpinBox.value()
                                    )
            logger.info(result)
            self.display_result(result)
        except Exception as e:
            logger.error(f"推理失败: {e}")
    
    def display(self, frame):
        # 创建一个与frame大小相同的全黑蒙版
        overlay = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # 如果有ROI区域，在结果图像上绘制ROI
        if hasattr(self, 'current_roi') and self.current_roi is not None:
            # 绘制ROI多边形
            if len(self.current_roi) >= 3:
                # 转换ROI为numpy数组
                pts = np.array(self.current_roi, np.int32)
                pts = pts.reshape((-1, 1, 2))

                # 填充多边形（红色，半透明）
                cv2.fillPoly(overlay, [pts], color=(255, 0, 0))  # RGB，红色

                # 融合overlay到原图，alpha控制透明度
                alpha = 0.3  # 透明度 0~1
                overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # 绘制红色实线边界
                cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # 绘制ROI点
            for pt in self.current_roi:
                cv2.circle(overlay, (pt[0], pt[1]), 3, (0, 0, 255), -1)

        h, w, ch = overlay.shape
        qimg = QImage(overlay, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 按 QLabel 大小缩放，但保持比例
        scaled_pixmap = pixmap.scaled(
            self.ui.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.label.setPixmap(scaled_pixmap)

    def display_result(self, result_frame):
        frame, details = self.drawer.run(self.frame, result_frame)

        self.display(frame)

        for index, instance in enumerate(details):
            label = QLabel()
            column_width = self.ui.tableWidget.columnWidth(3)
            row_height = self.ui.tableWidget.rowHeight(index)
            label.setFixedSize(column_width, row_height)

            instance_image = instance['image']
            instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
            instance_class = instance['class']
            instance_score = instance['socre']
            instance_coordinate = instance['coordinate']

            height, width, channel = instance_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(instance_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(label.size(),
                                   Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation
                                )
            label.setPixmap(pixmap)

            instance_class = QTableWidgetItem(instance_class)
            instance_class.setTextAlignment(Qt.AlignCenter)
            self.ui.tableWidget.setItem(index, 0, instance_class)

            instance_score = QTableWidgetItem(instance_score)
            instance_score.setTextAlignment(Qt.AlignCenter)
            self.ui.tableWidget.setItem(index, 1, instance_score)
            self.ui.tableWidget.setItem(index, 2, QTableWidgetItem(instance_coordinate))
            self.ui.tableWidget.setCellWidget(index, 3, label)
        
        # 如果是文件夹模式，显示导航信息
        if hasattr(self.stream, 'is_video') and not self.stream.is_video and hasattr(self.stream, 'image_files'):
            current_file = os.path.basename(self.stream.image_files[self.stream.current_index])
            total_files = len(self.stream.image_files)
            logger.info(f"当前图片: {current_file} ({self.stream.current_index + 1}/{total_files})")

    def next_image(self):
        """移动到下一张图片（文件夹模式）"""
        if (hasattr(self.stream, 'is_video') and not self.stream.is_video and 
            hasattr(self.stream, 'next') and self.stream.next()):
            # 读取下一张图片
            self.frame = self.stream.read()
            if self.frame is not None:
                self.display(self.frame)
                
                # 自动进行推理
                self.process_current_frame()
            else:
                logger.warning("已到达文件夹末尾")
        else:
            logger.warning("当前不是文件夹模式或无法移动到下一张图片")

    def set_roi_image(self):
        """设置ROI（感兴趣区域）- 允许用户在图像上绘制多边形围栏"""
        self.redraw_with_roi()

        if not hasattr(self, 'roi_mode') or not self.roi_mode:
            # 进入ROI绘制模式
            self.roi_mode = True
            self.drawing_roi = False
            
            # 更改鼠标样式，提示用户可以绘制
            self.ui.label.setCursor(Qt.CrossCursor)
            
            # 绑定鼠标事件到label
            self.ui.label.mousePressEvent = self.roi_mouse_press
            self.ui.label.mouseMoveEvent = self.roi_mouse_move
            self.ui.label.mouseDoubleClickEvent = self.roi_mouse_double_click
            
            # 更新按钮状态
            self.ui.setROIButton.setText("完成绘制")
            self.ui.setROIButton.setStyleSheet("background-color: #4CAF50; color: white;")
            
            logger.info('进入ROI绘制模式，点击图像绘制多边形，双击完成绘制')
        else:
            # 完成ROI绘制
            self.finish_roi_drawing()
    
    def roi_mouse_press(self, event):
        """ROI绘制时的鼠标按下事件"""
        if not self.roi_mode:
            return
            
        if event.button() == Qt.LeftButton:
            if not self.drawing_roi:
                self.roi_points = []
                self.drawing_roi = True
            
            # 获取鼠标位置
            pos = event.pos()
            img_x, img_y = self.map_label_to_image(pos)
            self.roi_points.append((img_x, img_y))
            
            # 重绘图像显示ROI
            self.redraw_with_roi()
            logger.info(f'添加ROI点: ({img_x}, {img_y})')
    
    def roi_mouse_move(self, event):
        """ROI绘制时的鼠标移动事件"""
        if not self.roi_mode or not self.drawing_roi:
            return
            
        # 记录鼠标位置用于绘制预览线
        self.last_mouse_pos = event.pos()
        self.redraw_with_roi()
    
    def roi_mouse_double_click(self, event):
        """ROI绘制时的鼠标双击事件"""
        if not self.roi_mode or not self.drawing_roi:
            return
            
        if len(self.roi_points) >= 3:
            self.drawing_roi = False
            self.finish_roi_drawing()
            logger.info(f'完成ROI绘制，共{len(self.roi_points)}个点')
    
    def redraw_with_roi(self):
        """重绘图像，显示ROI多边形"""
        display_frame = self.frame.copy()
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # 绘制ROI多边形
        if hasattr(self, 'roi_points'):
            if len(self.roi_points) > 1:               
                # 绘制多边形
                if len(self.roi_points) >= 3:
                    # 转换ROI为numpy数组
                    pts = np.array(self.roi_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))

                    # 创建一个与frame大小相同的全黑蒙版
                    overlay = display_frame.copy()

                    # 填充多边形（红色，半透明）
                    cv2.fillPoly(overlay, [pts], color=(255, 0, 0))  # BGR，红色

                    # 融合overlay到原图，alpha控制透明度
                    alpha = 0.3  # 透明度 0~1
                    display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)

                    # 绘制红色实线边界
                    cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

                # 绘制点
                for pt in self.roi_points:
                    cv2.circle(display_frame, (pt[0], pt[1]), 3, (0, 0, 255), -1)
                
                # # 绘制预览线（如果正在绘制）
                # if self.drawing_roi and hasattr(self, 'last_mouse_pos') and self.roi_points:
                #     last_img_pt = self.map_label_to_image(self.roi_points[-1])
                #     current_img_pt = self.map_label_to_image(self.last_mouse_pos)
                #     cv2.line(display_frame, last_img_pt, current_img_pt, (0, 255, 255), 2)
        
        # 显示更新后的图像
        h, w, ch = display_frame.shape
        qimg = QImage(display_frame, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.ui.label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.ui.label.setPixmap(scaled_pixmap)
    
    def map_label_to_image(self, label_point):
        """将QLabel坐标映射到图像坐标"""
        if self.frame is None:
            return (0, 0)

        img_h, img_w = self.frame.shape[:2]
        label_w = self.ui.label.width()
        label_h = self.ui.label.height()

        # 计算图像缩放后显示的尺寸（保持比例）
        pixmap = self.ui.label.pixmap()
        if pixmap is None:
            return (0, 0)
        scaled_w = pixmap.width()
        scaled_h = pixmap.height()

        # 计算图像在 QLabel 中的偏移量（因为 KeepAspectRatio 居中显示）
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2

        # 判断点击点是否在图像区域内
        x_in_image = label_point.x() - offset_x
        y_in_image = label_point.y() - offset_y
        if x_in_image < 0 or y_in_image < 0 or x_in_image > scaled_w or y_in_image > scaled_h:
            return (0, 0)  # 点击在空白区域

        # 映射到原图坐标
        scale_x = img_w / scaled_w
        scale_y = img_h / scaled_h
        img_x = int(x_in_image * scale_x)
        img_y = int(y_in_image * scale_y)

        return (img_x, img_y)

    def finish_roi_drawing(self):
        """完成ROI绘制"""
        self.roi_mode = False
        self.drawing_roi = False
        
        # 恢复鼠标样式
        self.ui.label.setCursor(Qt.ArrowCursor)
        
        # 恢复按钮状态
        self.ui.setROIButton.setText("设置视频围栏")
        self.ui.setROIButton.setStyleSheet("")
        
        # 保存ROI区域信息
        if hasattr(self, 'roi_points') and len(self.roi_points) >= 3:
            self.current_roi = self.roi_points
            logger.info(f'ROI设置完成，区域包含{len(self.roi_points)}个点')
        else:
            self.current_roi = None
            logger.info('ROI绘制取消或点数不足')
        
        # 移除临时鼠标事件绑定
        self.ui.label.mousePressEvent = None
        self.ui.label.mouseMoveEvent = None
        self.ui.label.mouseDoubleClickEvent = None

    def clear_roi_image(self):
        """清除ROI（感兴趣区域）"""
        # 清除ROI相关状态
        if hasattr(self, 'roi_mode'):
            self.roi_mode = False
        if hasattr(self, 'drawing_roi'):
            self.drawing_roi = False
        if hasattr(self, 'roi_points'):
            self.roi_points = []
        if hasattr(self, 'current_roi'):
            self.current_roi = None
        
        # 恢复鼠标样式
        if hasattr(self, 'ui') and hasattr(self.ui, 'label'):
            self.ui.label.setCursor(Qt.ArrowCursor)
            
            # 移除临时鼠标事件绑定
            self.ui.label.mousePressEvent = None
            self.ui.label.mouseMoveEvent = None
            self.ui.label.mouseDoubleClickEvent = None
        
        # 恢复按钮状态
        if hasattr(self, 'ui') and hasattr(self.ui, 'setROIButton'):
            self.ui.setROIButton.setText("设置视频围栏")
            self.ui.setROIButton.setStyleSheet("")
        
        # 如果有原始帧，重新显示原始图像
        if hasattr(self, 'frame') and self.frame is not None:
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.ui.label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.ui.label.setPixmap(scaled_pixmap)
        
        logger.info('ROI区域已清除')

    def clear_image(self):
        if self.frame is not None:
            self.ui.label.setStyleSheet("background-color: white;")
            self.display(self.frame)

    def export_data(self):
        logger.info('export_data ok!')