import os, sys
# Setting CUDA_VISIBLE_DEVICES is fine, but ROS needs to be imported first
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import copy
import numpy as np
import torch
from PIL import Image as PILImage
import matplotlib.pyplot as plt

# Grounding DINO Imports
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict, load_image

# Segment Anything Imports
from segment_anything import build_sam, SamPredictor, build_sam_vit_b
import cv2
import supervision as sv
import groundingdino.datasets.transforms as T
import time
GDIN_TRANSFORM = T.Compose(
    [
        T.RandomResize([800], max_size=1333), 
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- NEW IMAGE PREPARATION FUNCTION ---

def prepare_image_from_np(cv_image: np.ndarray):
    """
    ROS-compatible function to convert BGR NumPy image into the format required 
    by GroundingDINO (RGB NumPy array and PyTorch Tensor).
    
    Args:
        cv_image: NumPy array (H, W, 3) in BGR format, typically from cv_bridge.
        
    Returns:
        tuple: (image_source_rgb_np, image_transformed_tensor) 
    """
    # 1. Convert BGR NumPy to RGB NumPy (needed for PIL/SAM/Visualization)
    image_source_rgb_np = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # 2. Convert to PIL Image for the standard GroundingDINO transform pipeline
    image_source_pil = PILImage.fromarray(image_source_rgb_np)
    
    # 3. Apply the fixed GroundingDINO transforms
    # Returns tensor and target (which is None here)
    image_transformed_tensor, _ = GDIN_TRANSFORM(image_source_pil, None) 
    
    return image_source_rgb_np, image_transformed_tensor 
# --- MODEL LOADING (KEEP AS IS) ---
def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"GroundingDINO Model Load Result: {load_res}")
    _ = model.eval()
    return model

# --- MASK VISUALIZATION (KEEP AS IS) ---
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = PILImage.fromarray(image).convert("RGBA")
    mask_image_pil = PILImage.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(PILImage.alpha_composite(annotated_frame_pil, mask_image_pil))
# ----------------------------------------

class GroundedSAMNode(Node):
    def __init__(self):
        super().__init__('grounded_sam_processor')
        self.declare_parameter('image_topic', '/rgb')
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')

        # **1. Initialize CV Bridge and Publisher**
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10)
        
        # Publisher for the final annotated image
        self.publisher_ = self.create_publisher(Image, 'grounded_sam/segmentation', 10)

        # **2. Model and Hyperparameter Setup**
        ckpt_filename = "/home/yukang/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        ckpt_config_filename = "/home/yukang/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.groundingdino_model = load_model(ckpt_config_filename, ckpt_filename)

        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load SAM
        sam_checkpoint = '/home/yukang/GroundingDINO/weights/sam_vit_b_01ec64.pth'
        sam = build_sam_vit_b(checkpoint=sam_checkpoint)
        sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(sam)

        # Detection Thresholds
        self.TEXT_PROMPT = "ground"
        self.BOX_TRESHOLD = 0.3
        self.TEXT_TRESHOLD = 0.25
        self.get_logger().info('Grounded-SAM Node Initialized.')

    def image_callback(self, msg: Image):
        start = time.time()
        # 1. ROS Image to CV Array (BGR format)
        try:
            # **Convert ROS Image message to BGR NumPy array**
            # Assuming bgr8 encoding is used by the camera, change if needed
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") 
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        # **2. Prepare Image using GroundingDINO's utility (accepting NumPy)**
        # This function (aliased as load_image_np) takes the BGR OpenCV image and 
        # returns the RGB NumPy array (image_source_rgb) for SAM/Visualization 
        # and the transformed PyTorch tensor (image_transformed_tensor) for GDIN.
        # This replaces the original file-based loading and transformation.
        try:
            image_source_rgb, image_transformed_tensor = prepare_image_from_np(cv_image) 
        except Exception as e:
            self.get_logger().error(f"Error preparing image for GroundingDINO: {e}")
            return
        print("Image Processing", time.time() - start)
        start = time.time()
        # 3. GroundingDINO Prediction (Detection)
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.groundingdino_model,
                image=image_transformed_tensor.to(self.DEVICE), 
                caption=self.TEXT_PROMPT,
                box_threshold=self.BOX_TRESHOLD,
                text_threshold=self.TEXT_TRESHOLD,
                device=self.DEVICE
            )
        
        # If no objects are found, publish the original image or skip
        if len(boxes) == 0:
            self.publisher_.publish(msg)
            return
        print("Dino", time.time() - start)
        start = time.time()
        # 4. Segment Anything (SAM) Prediction (Segmentation)
        # SAM takes the raw RGB NumPy array as input
        self.sam_predictor.set_image(image_source_rgb) 

        H, W, _ = image_source_rgb.shape
        # Convert normalized boxes (cxcywh) to absolute pixel coordinates (xyxy)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        # Apply SAM's specific transformation to the boxes
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source_rgb.shape[:2]
        ).to(self.DEVICE)
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False, # Get the single best mask
        )
        print("SAM", time.time() - start)
        start = time.time()
        # 5. Visualize and Publish Result
        
        # Use GroundingDINO's annotate function on the original BGR image for text labels
        annotated_frame_bgr = annotate(image_source=cv_image, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame_rgb = annotated_frame_bgr[..., ::-1] # BGR to RGB for mask overlay

        # Overlay all masks onto the annotated image
        final_annotated_frame = annotated_frame_rgb
        for mask in masks:
            mask_np = mask[0].detach().cpu().numpy()
            final_annotated_frame = show_mask(mask_np, final_annotated_frame)

        # Convert final RGB result back to BGR for ROS Image message
        final_annotated_frame_bgr = cv2.cvtColor(final_annotated_frame, cv2.COLOR_RGBA2BGR)
        
        # Publish the result
        try:
            output_msg = self.bridge.cv2_to_imgmsg(final_annotated_frame_bgr, encoding="bgr8")
            output_msg.header = msg.header
            self.publisher_.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")
        print("Final", time.time() - start)

def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAMNode()
    rclpy.spin(node)
    
    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
