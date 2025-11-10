import os
import cv2
import numpy as np
import torch
from PIL import Image
import pysaliency
from resmem import ResMem, transformer


# For SALICON, we'll use a simple placeholder or find a PyTorch impl. Here, assume a model load if available.
# Note: SALICON requires a deep model; for simplicity, use a pre-trained from somewhere or skip.
# For SIG, implement manually based on description.

class SaliencyComputer:
    def __init__(self):
        # Initialize pysaliency models
        self.gbvs = pysaliency.GBVS()
        self.itti = pysaliency.IKN()  # Itti Koch Niebur
        self.aim = pysaliency.AIM()
        self.sun = pysaliency.SUN()
        self.aws = pysaliency.AWS()
        self.bms = pysaliency.BMS()

        # For ResMem memorability
        self.resmem_model = ResMem(pretrained=True)
        self.resmem_model.eval()

        # For SALICON, need a deep model. Placeholder: Assume a function or load if available.
        # You may need to install additional libs or models for SALICON.
        # For example, from OpenSALICON, but it uses Caffe. Alternatively, use a PyTorch saliency model.
        # Here, skip or implement a dummy.
        self.salicon = None  # Replace with actual model if available

        # For SIG (Image Signature)
        # Implement based on the method: sign of DCT coefficients, then inverse.

    def compute_sig(self, image):
        # Image Signature: sign of DCT, then inverse DCT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        sig = np.sign(dct)
        recon = cv2.idct(sig)
        sal_map = np.abs(recon) ** 2  # Square for saliency
        sal_map = cv2.normalize(sal_map, None, 0, 255, cv2.NORM_MINMAX)
        return sal_map.astype(np.uint8)

    def compute_salicon(self, frame):
        # Placeholder for SALICON. Implement if you have the model.
        # For example, using a deep network.
        return np.zeros_like(frame[:, :, 0])  # Dummy

    def compute_resmem(self, frame):
        # ResMem for memorability (scalar per image)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            mem_score = self.resmem_model(transformer(pil_img).unsqueeze(0))
        return mem_score.item()  # Scalar, not map

    def compute_saliency_maps(self, video_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Compute for each model
            models = {
                'gbvs': self.gbvs,
                'itti': self.itti,
                'aim': self.aim,
                'sun': self.sun,
                'aws': self.aws,
                'bms': self.bms,
            }

            for model_name, model in models.items():
                sal_map = model.saliency_map(frame)  # pysaliency models take image array
                sal_map = cv2.normalize(sal_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f'{model_name}_frame_{frame_idx}.png'), sal_map)

            # SIG
            sig_map = self.compute_sig(frame)
            cv2.imwrite(os.path.join(output_dir, f'sig_frame_{frame_idx}.png'), sig_map)

            # SALICON (placeholder)
            salicon_map = self.compute_salicon(frame)
            cv2.imwrite(os.path.join(output_dir, f'salicon_frame_{frame_idx}.png'), salicon_map)

            # ResMem (memorability score, save as text)
            mem_score = self.compute_resmem(frame)
            with open(os.path.join(output_dir, f'resmem_frame_{frame_idx}.txt'), 'w') as f:
                f.write(str(mem_score))

            frame_idx += 1

        cap.release()
        print(f"Processed {frame_idx} frames. Outputs in {output_dir}")

# Usage example
# computer = SaliencyComputer()
# computer.compute_saliency_maps('path/to/video.mp4', 'output_directory')