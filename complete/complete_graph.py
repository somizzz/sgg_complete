import os
import h5py
import numpy as np
import json
import torch
import torchvision
import torch.nn.functional as F
import shutil
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import cv2
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 配置类，定义路径和其他参数
class Config:
    def __init__(self):
        self.vg_image_dir = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/stanford_spilt/VG_100k_images"  # 图像目录
        self.detector_weights = ""  # 检测器权重
        self.output_dir = "/home/p_zhuzy/p_zhu/PySGG-main/ouotput"  # 输出目录
        self.roidb_path = os.path.join(self.output_dir, "roidb.npz")  # 检测结果保存路径
        self.node_feat_dir = os.path.join(self.output_dir, "node_features")  # 特征保存目录
        self.graph_dir = os.path.join(self.output_dir, "graphs")  # 图数据保存目录
        self.confidence_threshold = 0.5  # 置信度阈值
        self.top_k_ratios = [0.6, 0.8]  # 图构建中保留边的比例
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.node_feat_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)

class VGDetector:
    def __init__(self, config):
        """Initialize the detector with a given configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_detector()

    def _load_detector(self):
        """Load and configure the Faster R-CNN model with ResNet-50-FPN backbone."""
        print("Loading Faster R-CNN detector with ResNet-50-FPN backbone...")
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        model = FasterRCNN(backbone=backbone, num_classes=91)  # 91 classes (e.g., COCO)
        
        if hasattr(self.config, 'detector_weights') and self.config.detector_weights:
            weights_path = self.config.detector_weights
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            print("Checkpoint loaded from:", weights_path)
            state_dict = checkpoint.get('model', checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            try:
                model.load_state_dict(state_dict)
                print("Weights loaded successfully.")
            except RuntimeError as e:
                print(f"Error loading weights: {e}")
                model.load_state_dict(state_dict, strict=False)
                print("Weights loaded with strict=False; some keys may be mismatched.")
        
        model.to(self.device)
        model.eval()
        print("Detector loaded and ready.")
        return model

    def detect(self, image_path):
        """Detect objects in an image and return bounding boxes, labels, scores, and ROI features."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}. Check if the file exists and is readable.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torchvision.transforms.ToTensor()(image).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        rois = predictions["boxes"].cpu().numpy()
        
        return boxes, labels, scores, rois

class VGFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.visual_dim = 2048
        self.word_dim = 768
        self.spatial_dim = 8
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").eval()
        self.idx_to_label = self._load_idx_to_label()
        self.detector = VGDetector(config)
        #self.mlp = self._build_mlp()

    def _build_mlp(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.visual_dim + self.word_dim + self.spatial_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1024)
        )

    def _load_idx_to_label(self):
        dict_file = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-dicts-with-attri.json"
        with open(dict_file, 'r') as f:
            data = json.load(f)
        return data.get('idx_to_label', {})

    def _get_label_name(self, label_idx):
        return self.idx_to_label.get(str(label_idx), "Unknown")

    def _get_spatial_feature(self, box, img_size):
        h, w = img_size
        x1, y1, x2, y2 = box
        return np.array([x1/w, y1/h, x2/w, y2/h, (x2-x1)/w, (y2-y1)/h, ((x1+x2)/2)/w, ((y1+y2)/2)/h])

    def extract(self, image_path, boxes, labels,rois):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path} in feature extraction.")
        img_size = img.shape[:2]
        #boxes, labels, scores, visual_features = self.detector.detect(image_path)
        node_features = []
        for i, (box, label) in enumerate(zip(boxes, labels)):
            v_feat = rois[i].flatten()
            label_name = self._get_label_name(label)
            inputs = self.tokenizer(label_name, return_tensors="pt")
            with torch.no_grad():
                w_feat = self.bert(**inputs).last_hidden_state[:, 0, :].numpy()[0]
            s_feat = self._get_spatial_feature(box, img_size)
            combined = np.concatenate([v_feat, w_feat, s_feat])
            #node_feat = self.mlp(torch.FloatTensor(combined)).detach().numpy()
            node_features.append(combined)
        return np.array(node_features)

class GraphConstructor:
    def __init__(self, config):
        self.config = config

    def build_graphs(self, node_features):
        if len(node_features) < 2:
            return {f"G_{int(ratio*100)}": {"nodes": node_features, "edges": [], "weights": []} for ratio in self.config.top_k_ratios}
        sim_matrix = cosine_similarity(node_features)
        n = len(node_features)
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        similarities = [sim_matrix[i,j] for i, j in edges]
        sorted_idx = np.argsort(similarities)[::-1]
        edges = np.array(edges)[sorted_idx]
        similarities = np.array(similarities)[sorted_idx]
        graphs = {}
        for ratio in self.config.top_k_ratios:
            keep_num = int(len(edges) * ratio)
            graphs[f"G_{int(ratio*100)}"] = {
                "nodes": node_features,
                "edges": edges[:keep_num],
                "weights": similarities[:keep_num]
            }
        return graphs

# 主程序
if __name__ == "__main__":
    cfg = Config()

    # 阶段1: 数据划分
    image_dir = cfg.vg_image_dir
    all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    train_images, test_images = train_test_split(all_image_files, test_size=0.2, random_state=42)
    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print("DEBUG: Starting to copy training images...")
    for img in tqdm(train_images, desc="Copying train images"):
        shutil.copy(os.path.join(image_dir, img), train_dir)
    print(f"DEBUG: Copied {len(train_images)} training images.")
    
    print("DEBUG: Starting to copy test images...")
    for img in tqdm(test_images, desc="Copying test images"):
        shutil.copy(os.path.join(image_dir, img), test_dir)
    print(f"DEBUG: Copied {len(test_images)} test images.")

    # 阶段2: 物体检测
    detector = VGDetector(cfg)
    roidb = {"train": [], "test": []}
    for split in ["train", "test"]:
        split_dir = os.path.join(image_dir, split)
        img_names = os.listdir(split_dir)
        print(f"DEBUG: Detecting objects in {split} set with {len(img_names)} images...")
        for img_name in tqdm(img_names, desc=f"Detecting {split} set"):
            img_path = os.path.join(split_dir, img_name)
            try:
                boxes, labels, scores, rois = detector.detect(img_path)
                roidb[split].append({
                    "image": img_path, 
                    "boxes": boxes, 
                    "labels": labels, 
                    "scores": scores, 
                    "rois": rois})
            except ValueError as e:
                print(f"Warning: {e} - Skipping image: {img_path}")
                continue
    np.savez(cfg.roidb_path, **roidb)
    print(f"DEBUG: Object detection completed, results saved to {cfg.roidb_path}")

    # 阶段3: 特征提取
    fe = VGFeatureExtractor(cfg)
    for split in ["train", "test"]:
        image_features = []
        print(f"DEBUG: Extracting features for {split} set with {len(roidb[split])} images...")
        for entry in tqdm(roidb[split], desc=f"Extracting features ({split})"):
            try:
                features = fe.extract(
                    entry["image"], 
                    entry["boxes"], 
                    entry["labels"],
                    entry["rois"]
                    )
                image_features.append(features)
            except ValueError as e:
                print(f"Warning: {e} - Skipping feature extraction for image: {entry['image']}")
                continue
        feature_path = os.path.join(cfg.node_feat_dir, f"{split}_features.npy")
        np.save(feature_path, image_features)
        print(f"DEBUG: Features for {split} set saved to {feature_path}")

    # 阶段4: 图构建
    gc = GraphConstructor(cfg)
    for split in ["train", "test"]:
        feature_path = os.path.join(cfg.node_feat_dir, f"{split}_features.npy")
        image_features = np.load(feature_path, allow_pickle=True)  
        print(f"DEBUG: Building graphs for {split} set with {len(image_features)} images...")
    
        per_image_graphs = []  # 每次循环split时重新初始化，避免累积
        for features in tqdm(image_features, desc=f"Building graphs ({split})"):
            graphs = gc.build_graphs(features)  # 生成单张图片的图数据
            per_image_graphs.append(graphs)  # 按图片保存
    
        graph_path = os.path.join(cfg.graph_dir, f"{split}_graphs.npz")
        np.savez(graph_path, graphs=per_image_graphs)  
        print(f"DEBUG: Graphs for {split} set saved to {graph_path}")