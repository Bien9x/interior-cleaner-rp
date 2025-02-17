import os
import time
from typing import List

import torch

from models.inpaint.lama import Lama
from models.diffusion.sdxl import SDXLControlnetInpaint
from models.prompting.wd_tagger import TagGenerator

from PIL import Image

def test_lama():
    print('Loading model...')
    lama = Lama()
    lama.setup()
    print('Loading image...')
    image = Image.open('tmp/tulanh_hd.jpg')
    mask = Image.open('tmp/tulanh_mask.png')
    print(f'Image shape: {image.size} - {mask.size}')
    print('Running lama...')
    image_lama = lama(image, mask)
    image_lama.save('tmp/tulanh_lama.png')
    print('Done')

def test_prompt_generator():
    print('Loading model...')
    prompter = TagGenerator()
    prompter.setup()
    print('Loading image...')
    image = Image.open('tmp/tulanh_lama.png')
    print('Generating prompt...')
    prompt = prompter(image)
    print(prompt)

def test_repaint():
    print('Loading model...')
    repainter = SDXLControlnetInpaint()
    repainter.setup()
    print('Loading image...')
    image = Image.open('tmp/tulanh_lama.png')
    mask = Image.open('tmp/tulanh_mask.png')
    prompt = 'no humans, indoors, scenery, window, ceiling light, curtains, ceiling, building, apartment, bed'
    print(f'Image shape: {image.size} - {mask.size}')
    print('Running diffusion...')
    image_out = repainter(image, mask, prompt)
    image_out[0].save('tmp/tulanh_out.png')
    print('Done')

if __name__ == '__main__':
    #print('Testing lama...')
    #test_lama()
    # print('Testing prompt...')
    # test_prompt_generator()
    print('Testing diffusion...')
    test_repaint()