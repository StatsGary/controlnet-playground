# ControlNet Playground
![Python Versions](https://img.shields.io/pypi/pyversions/controlnet)
![Profile View Counter](https://komarev.com/ghpvc/?username=StatsGary)
![Last Commit](https://img.shields.io/github/last-commit/StatsGary/controlnet_playground)
![Stars](https://img.shields.io/github/stars/StatsGary/controlnet_playground?style=social)

The supporting repository for all things controlnet. 

## Getting setup

**This project was built in `Python 3.9` and requires the requirements file to be installed.**

1. Firstly, clone this repository, as you will need this for the class files to work in your project. To do this use: 

```
git clone https://github.com/StatsGary/controlnet_playground.git
```

2. Next, you will need to install your package dependencies. I would recommend using a seperate virtual environment for the installation:

```python
pip install -r requirements.txt
```

3. Once these packages are installed, then you are good to follow on with the tutorials in the next sections.

## Living room remodeller

![](images/livingroom.gif)

Living room remodeller - a model that uses semantic segmentation and MLSD edge detection to take an input of a room and generate what it thinks your living room should look like, based on a prompt. 

Check out this post for details of what this does: https://hutsons-hacks.info/using-controlnet-models-to-remodel-my-living-room. 

To use the remodeller, copy the class from the article in Python, and then created a `main.py` file, or encapsulate in main block, as below: 

```python
# Import our custom classes from this repo
from controlnet.remodeller import ControlNetMLSD, ControlNetSegment

if __name__=='__main__':
    prompt = 'living room with navy theme'
    img_path = 'images/house.jpeg'

    # Run the MLSD edge detector version
    mlsd_net_seg = ControlNetMLSD(
        prompt=prompt, 
        image_path=img_path
    )
    
    mlsd_net_seg.generate_mlsd_image(
        mlsd_save_path=f'images/house_mlsd_{prompt.strip().replace(" ", "")}.jpeg',
        mlsd_diff_gen_save_path=f'images/house_mlsd_gen_{prompt.strip().replace(" ", "")}.jpeg'
        )

    # Run the semantic segmentation model
    control_net_seg = ControlNetSegment(
        prompt=prompt,
        image_path=img_path)
    
    seg_image = control_net_seg.segment_generation(
        save_segmentation_path=f'images/house_seg_{prompt.strip().replace(" ", "")}.jpeg',
        save_gen_path=f'images/house_seg_gen_{prompt.strip().replace(" ", "")}.jpeg'
        )
```

## Doodle Face

![](images/MiniDoofdle.gif)

Doodle face - a model to take a profile picture and convert into your favourite animated images and some historical figures. 

See the supporting post: https://hutsons-hacks.info/creating-doodles-with-hed-detection-and-controlnet.

To use this model, refer to the blog post, or import the class from this repository:

``` python
# Import custom installs
from controlnet.scribble_net import ScribbleControlNet

if __name__=='__main__':
    # Class instance
    doodle = ScribbleControlNet(
        'images/man.jpeg'
    )
    print(doodle)
    
    # Create the prompt
    prompt = "monster"
    
    # Generate the image
    image_gen = doodle.generate_scribble(prompt, 
                                           num_inf_steps=50,
                                           save_path=f'images/{prompt.strip().replace(" ", "")}')
```



