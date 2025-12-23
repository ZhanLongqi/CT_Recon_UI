import numpy as np
import dearpygui.dearpygui as dpg
import traceback
class Texture_registry:
    def __init__(self,tag):
        dpg.texture_registry(tag=tag)
        self.tag = tag

    def add_texture(self, width, height,default_value=None,tag=None):

        if default_value is None:
            default_value = np.zeros((height, width, 4), dtype=np.float32).flatten()
        try:
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    format=dpg.mvFormat_Float_rgba,
                    default_value=default_value,
                    parent=self.tag,
                    tag=tag,
                )
        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"❌ Error creating texture {tag}: {str(e)}")
            print(f"📌 Error details (file/line/function):\n{err_trace}")

def create_texture_registry(my_data):

    my_texture_registry = Texture_registry(tag="__texture_container") 
    my_texture_registry.add_texture(
        width=my_data['proj_width'],
        height=my_data['proj_height'],
        tag="raw_proj"
    )
    my_texture_registry.add_texture(
        width=my_data['proj_width'],
        height=my_data['proj_height'],
        tag="attenuation_proj"
    )
    my_texture_registry.add_texture(
        width=my_data['proj_width'],
        height=my_data['proj_width'],
        tag='recon_slice'
    )
    my_texture_registry.add_texture(
        width=my_data['proj_width'],
        height=my_data['proj_width'],
        tag='recon_slice_dering'
    )   

