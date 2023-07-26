from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image


class ControlNetSegment:
    def __init__(self, prompt, image_path, image_processor='openmmlab/upernet-convnext-small',
                image_segment="openmmlab/upernet-convnext-small", pretrain_control_net="fusing/stable-diffusion-v1-5-controlnet-seg",
                pretrain_stable_diffusion="runwayml/stable-diffusion-v1-5"):
        # Set our class variables
        self.image_path = image_path
        # Get the pretrained models from HuggingFace
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor)
        self.image_segment = UperNetForSemanticSegmentation.from_pretrained(image_segment)
        self.control_net = pretrain_control_net
        self.stable_diffusion = pretrain_stable_diffusion
        self.prompt = prompt
        
        # Check the CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If GPU not available in torch then throw error up instantiation
        if self.device == 'cpu':
            raise MemoryError('GPU needed for inference in this project')
        
        # Raise error if assert statement is not met
        assert isinstance(image_path, str), 'Image path must be a string linking to an image'


    def segment_generation(self, 
                           save_segmentation_path=None, 
                           save_gen_path=None, 
                           num_inf_steps=50):
        # Set variable
        self.num_inf_steps = num_inf_steps
        # Convert the image
        image = Image.open(self.image_path).convert('RGB')
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segment(pixel_values)
            seg = self.image_processor.post_process_semantic_segmentation(outputs, 
                                                                  target_sizes=[image.size[::-1]])[0]
            
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]])
        
        # Loop through the colour palette
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color  

        # Create the semantic segmentation of the room
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        # Check if there is a value in the path then save the segmentation to path
        if save_segmentation_path is not None:
            image.save(save_segmentation_path)

        # Load pretrained model 
        controlnet = ControlNetModel.from_pretrained(
            self.control_net, torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.stable_diffusion, controlnet=controlnet, safety_checker=None, 
            torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        image = pipe(self.prompt, 
                     image, 
                     num_inference_steps=self.num_inf_steps).images[0]
        
        if save_gen_path is not None:
            image.save(save_gen_path)
        return image
    
class ControlNetMLSD:
    def __init__(self,prompt, image_path, mlsd_net='lllyasviel/ControlNet',
                 pretrain_control_net='fusing/stable-diffusion-v1-5-controlnet-mlsd',
                 stable_diffusion_model='runwayml/stable-diffusion-v1-5'):
        
        self.mlsd_net = mlsd_net
        self.image_path = image_path
        self.prompt = prompt
        self.mlsd = MLSDdetector.from_pretrained(self.mlsd_net)
        self.sd_model = stable_diffusion_model
        self.pretrain_control_net = pretrain_control_net

        # Check the CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If GPU not available in torch then throw error up instantiation
        if self.device == 'cpu':
            raise MemoryError('GPU needed for inference in this project')
        
        assert isinstance(image_path, str), 'Image path must be a string linking to an image'

    def generate_mlsd_image(self,num_inf_steps=50, mlsd_save_path=None, mlsd_diff_gen_save_path=None):
        self.num_inf_steps = num_inf_steps
        self.mlsd_image_path = mlsd_save_path
        self.mlsd_gen_save_path = mlsd_diff_gen_save_path

        image_loaded = load_image(self.image_path)
        mlsd_image = self.mlsd(image_loaded)

        if self.mlsd_image_path is not None:
            # Checks if there is a path to save the image
            mlsd_image.save(self.mlsd_image_path)
            print(f'[IMAGE SAVE INFO] MLSD image saved to path specified:{self.mlsd_image_path}')
        
        # Load in control net and pipe from HuggingFace
        controlnet = ControlNetModel.from_pretrained(
            self.pretrain_control_net, torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model, controlnet=controlnet, 
            safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        prompt = self.prompt
        print(f'Generating image based on prompt:\n{self.prompt}')
        image = pipe(prompt, mlsd_image, num_inference_steps=self.num_inf_steps).images[0]

        if self.mlsd_gen_save_path is not None:
            image.save(self.mlsd_gen_save_path)
            print(f"[IMAGE SAVE INFO] MLSD generated on '{self.prompt}' prompt saved to path specified:{self.mlsd_image_path}")

        return image
