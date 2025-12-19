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
        sino,
        sigma=dpg.get_value('Fitting_param_sigma'),
        order=dpg.get_value('Fitting_param_order'),
        sort=dpg.get_value('Fitting_param_sort'),
        num_chunk=dpg.get_value('Fitting_param_num_chunk')
    )

    return sino_clean

def Filtering_dering(sino):
    sino_clean = rem.remove_stripe_based_filtering(
        sino,
        sigma=dpg.get_value('Filtering_param_sigma'),
        size=dpg.get_value('Filtering_param_size'),
        sort=dpg.get_value('Filtering_param_sort')
    )

    return sino_clean

def Sorting_dering(sino):
    sino_clean = rem.remove_stripe_based_sorting(
    sino, 
    option={"method": "gaussian_filter", "para1": (dpg.get_value('Sorting_param_sigma'), dpg.get_value('Sorting_param_size'))}  # 高斯滤波，sigma=1, size=21
    )

    return sino_clean