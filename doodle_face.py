# Futures
from __future__ import unicode_literals
from __future__ import print_function
# Owned
__project__ = 'Doodle Face'
__author__ = "Gary Hutson"
__copyright__ = "Copyright 2023, Hutsons-hacks.info"
__credits__ = ["Gary Hutson"]
__license__ = "MPL 2.0"
__version__ = "0.1.0"
__maintainer__ = "Gary Hutson"
__email__ = "hutsons-hacks@outlook.com"
__status__ = "Dev"

# Imports

from controlnet.scribble_net import ScribbleControlNet

if __name__=='__main__':

    # Create header print before running code
    prompt = "alien"
    print('-' * 80)
    print(f'Script created by {__author__}\nProject: {__project__}')
    print(f'Based on prompt: {prompt}')
    print('-' * 80)

    
    doodle = ScribbleControlNet(
        'images/man.jpeg'
    )
    print(doodle)
    image_gen = doodle.generate_scribble(prompt, 
                                           num_inf_steps=50,
                                           save_path=f'images/{prompt.strip().replace(" ", "")}')
    
  