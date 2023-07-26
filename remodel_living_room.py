# Futures
from __future__ import unicode_literals
from __future__ import print_function
# Owned
__author__ = "Gary Hutson"
__copyright__ = "Copyright 2023, Hutsons-hacks.info"
__credits__ = ["Gary Hutson"]
__license__ = "MPL 2.0"
__version__ = "0.1.0"
__maintainer__ = "Gary Hutson"
__email__ = "hutsons-hacks@outlook.com"
__status__ = "Dev"

# Imports
# Custom imports from controlnet class
from controlnet.remodeller import ControlNetMLSD, ControlNetSegment

if __name__=='__main__':
    prompt = 'living room with navy theme'
    img_path = 'images/house.jpeg'

    mlsd_net_seg = ControlNetMLSD(
        prompt=prompt, 
        image_path=img_path
    )
    
    mlsd_net_seg.generate_mlsd_image(
        mlsd_save_path=f'images/house_mlsd_{prompt.strip().replace(" ", "")}.jpeg',
        mlsd_diff_gen_save_path=f'images/house_mlsd_gen_{prompt.strip().replace(" ", "")}.jpeg'
        )

    control_net_seg = ControlNetSegment(
        prompt=prompt,
        image_path=img_path)
    
    seg_image = control_net_seg.segment_generation(
        save_segmentation_path=f'images/house_seg_{prompt.strip().replace(" ", "")}.jpeg',
        save_gen_path=f'images/house_seg_gen_{prompt.strip().replace(" ", "")}.jpeg'
        )
    
  