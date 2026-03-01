import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define the classes we want to train YOLO on, and map them to their template images
# max_instances: how many of this element can appear on screen at once
CLASS_MAPPING = {
    0:  {"name": "start_btn",        "templates": ["点击开始.png"],                     "max": 1},
    1:  {"name": "close_btn",        "templates": ["游戏内很多页面窗口的叉.png", "内嵌公告的叉.png", "公告叉叉.png"], "max": 3},
    2:  {"name": "confirm_btn",      "templates": ["确认(可以点space）.png"],            "max": 1},
    3:  {"name": "cafe_earnings_btn", "templates": ["咖啡厅收益按钮.png"],               "max": 1},
    4:  {"name": "headpat_bubble",   "templates": ["可摸头的标志.png"],                  "max": 5},
    5:  {"name": "cafe_btn",         "templates": ["咖啡厅.png"],                       "max": 1},
    6:  {"name": "schedule_btn",     "templates": ["课程表.png"],                       "max": 1},
    7:  {"name": "home_btn",         "templates": ["Home按钮.png"],                     "max": 1},
    8:  {"name": "back_btn",         "templates": ["返回按钮.png"],                     "max": 1},
    9:  {"name": "left_switch",      "templates": ["左切换.png"],                       "max": 1},
    10: {"name": "right_switch",     "templates": ["右切换.png"],                       "max": 1},
    11: {"name": "club_btn",         "templates": ["社交.png"],                         "max": 1},
}

class AutoAnnotator:
    def __init__(self, templates_dir: str = "data/captures", threshold: float = 0.80):
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.loaded_templates = {}  # class_id -> [(name, bgr, mask_or_None, max_instances)]
        
        for class_id, info in CLASS_MAPPING.items():
            entries = []
            for tmpl_name in info["templates"]:
                tmpl_path = self.templates_dir / tmpl_name
                if not tmpl_path.exists():
                    print(f"Warning: Template {tmpl_name} not found.")
                    continue
                raw = cv2.imdecode(np.fromfile(str(tmpl_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    print(f"Failed to load {tmpl_name}")
                    continue
                if raw.ndim == 3 and raw.shape[2] == 4:
                    bgr = raw[:, :, :3]
                    mask = raw[:, :, 3]
                else:
                    bgr = raw
                    mask = None
                entries.append((tmpl_name, bgr, mask, info.get("max", 1)))
            self.loaded_templates[class_id] = entries

    def match_template_fast(self, image: np.ndarray, bgr_tmpl: np.ndarray,
                            mask: np.ndarray, max_instances: int) -> list:
        """
        Fast template matching using minMaxLoc + iterative suppression.
        Instead of np.where (millions of points), we only find the top N best matches.
        """
        h, w = bgr_tmpl.shape[:2]

        if mask is not None:
            res = cv2.matchTemplate(image, bgr_tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
        else:
            res = cv2.matchTemplate(image, bgr_tmpl, cv2.TM_CCOEFF_NORMED)

        matches = []
        for _ in range(max_instances):
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < self.threshold:
                break
            x, y = max_loc
            matches.append({"x": x, "y": y, "w": w, "h": h, "score": float(max_val)})
            # Suppress the matched region so we don't find it again
            x1 = max(0, x - w // 2)
            y1 = max(0, y - h // 2)
            x2 = min(res.shape[1], x + w // 2 + 1)
            y2 = min(res.shape[0], y + h // 2 + 1)
            res[y1:y2, x1:x2] = 0.0

        return matches

    def process_dataset(self, dataset_dir: str):
        dataset_dir = Path(dataset_dir)
        image_files = sorted(dataset_dir.glob("*.jpg"))
        
        print(f"Found {len(image_files)} images to annotate in {dataset_dir}")
        
        # Generate classes.txt
        with open(dataset_dir / "classes.txt", "w", encoding="utf-8") as f:
            for i in sorted(CLASS_MAPPING.keys()):
                f.write(f"{CLASS_MAPPING[i]['name']}\n")

        annotations_count = 0
        
        for img_path in tqdm(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            label_lines = []
            
            for class_id, entries in self.loaded_templates.items():
                for tmpl_name, bgr_tmpl, mask, max_inst in entries:
                    # Skip if template is larger than image
                    if bgr_tmpl.shape[0] > img_h or bgr_tmpl.shape[1] > img_w:
                        continue
                    matches = self.match_template_fast(img, bgr_tmpl, mask, max_inst)
                            
                    for m in matches:
                        x_center = (m["x"] + m["w"] / 2) / img_w
                        y_center = (m["y"] + m["h"] / 2) / img_h
                        norm_w = m["w"] / img_w
                        norm_h = m["h"] / img_h
                        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                        annotations_count += 1
            
            label_path = img_path.with_suffix('.txt')
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
                
        print(f"Auto-annotation complete! Generated {annotations_count} bounding boxes across {len(image_files)} images.")

if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_images/run_20260226_193214"
    annotator = AutoAnnotator(threshold=0.80)
    annotator.process_dataset(dataset_dir)
