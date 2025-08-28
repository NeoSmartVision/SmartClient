# core/result_saver.py

from utils.file_utils import save_yolo_result, save_voc_result

class ResultSaver:
    def __init__(self):
        pass

    def save_labels_batch(self, results, save_dir, format="YOLO", class_map=None):
        """
        批量保存推理结果为label文件
        :param results: {img_path: result_dict, ...}
        :param save_dir: 保存目录
        :param format: "YOLO" 或 "VOC"
        :param class_map: 类别映射字典
        :return: (success_count, fail_count, fail_list)
        """
        success_count = 0
        fail_count = 0
        fail_list = []
        for img_path, result in results.items():
            detections = result.get("data", {}).get("detections", [])
            try:
                if format == "YOLO":
                    save_yolo_result(img_path, detections, class_map, save_dir)
                else:
                    save_voc_result(img_path, detections, save_dir)
                success_count += 1
            except Exception as e:
                fail_count += 1
                fail_list.append((img_path, str(e)))
        return success_count, fail_count, fail_list