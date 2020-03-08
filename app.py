import time

import torch
import streamlit as st

from config import opt
from backend import cache_model_inputs_outputs


def image_display(cache_dict, input_scale = 0, output_scale = 9):
    st.title('Streamlit implementation if SinGAN')
    st.write("Here's our first attempt at implementing backend with streamlit integration for image display")
    imageLocation_input = st.empty()
    imageLocation_output = st.empty()
 
    for (i, o) in zip(cache_dict[input_scale]['input'], cache_dict[output_scale]['output']):
        imageLocation_input.image(i, channels = 'RGB')
        imageLocation_output.image(o, channels = 'RGB')
        time.sleep(0.3)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dict = cache_model_inputs_outputs(opt, device)
    #call function for front-end display
    image_display(cache_dict)