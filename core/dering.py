import numpy as np
import algotom.prep.removal as rem
import dearpygui.dearpygui as dpg
algorithms = [
    'None',
    'Sorting',
    'Filtering',
    'Fitting'
]
def dering(sinogram,algorithm,param):
    #https://github.com/algotom/algotom/blob/master/algotom/prep/removal.py
    match algorithm:
        case 'None':
            return sinogram
        case 'Sorting':
            return Sorting_dering(sinogram)
        case 'Filtering':
            return Filtering_dering(sinogram)
        case 'Fitting':
            return Fitting_dering(sinogram)

def Fitting_dering(sino):
    sino_clean = rem.remove_stripe_based_fitting(
        np.squeeze(sino),
        sigma=dpg.get_value('Fitting_param_sigma'),
        order=dpg.get_value('Fitting_param_order'),
        sort=dpg.get_value('Fitting_param_sort'),
        num_chunk=dpg.get_value('Fitting_param_num_chunk')
    )
    return np.expand_dims(sino_clean,axis=1)

def Filtering_dering(sino):
    sino_clean = rem.remove_stripe_based_filtering(
        np.squeeze(sino),
        sigma=dpg.get_value('Filtering_param_sigma'),
        size=dpg.get_value('Filtering_param_size'),
        sort=dpg.get_value('Filtering_param_sort')
    )
    return np.expand_dims(sino_clean,axis=1)

def Sorting_dering(sino):
    sino_clean = rem.remove_stripe_based_sorting(
    np.squeeze(sino), 
    option={"method": "gaussian_filter", "para1": (dpg.get_value('Sorting_param_sigma'), dpg.get_value('Sorting_param_size'))}  # 高斯滤波，sigma=1, size=21
    )
    return np.expand_dims(sino_clean,axis=1)