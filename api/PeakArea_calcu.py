import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from concurrent.futures import ProcessPoolExecutor

def calculate_peak_area(wave, spec, peak_wavenumber, peak_range=50):
    peak_index = np.argmin(np.abs(wave - peak_wavenumber))
    left_index = max(0, peak_index - peak_range)
    right_index = min(len(wave), peak_index + peak_range)

    wave_peak = wave[left_index:right_index]
    spec_peak = spec[left_index:right_index]

    def voigt(x, amp, cen, sigma, gamma):
        return amp * voigt_profile(x - cen, sigma, gamma)

    initial_guess = [np.max(spec_peak), peak_wavenumber, 1.0, 1.0]
    params, _ = curve_fit(voigt, wave_peak, spec_peak, p0=initial_guess, maxfev=100000)

    amp, cen, sigma, gamma = params
    area = amp * np.sqrt(np.pi) * sigma * (gamma / np.sqrt(sigma**2 + gamma**2))
    return area

def process_pixel(spec, wave, peak_wavenumber, peak_range):
    return calculate_peak_area(wave, spec, peak_wavenumber, peak_range)

def peak_area_highspec_image(data, wave, peak_wavenumber, peak_range=50):
    length, width, _ = data.shape
    
    data_flattened = data.reshape(-1, data.shape[-1])
    
    # 使用并行化来加速计算
    with ProcessPoolExecutor() as executor:
        areas = list(executor.map(process_pixel, data_flattened, [wave]*len(data_flattened), [peak_wavenumber]*len(data_flattened), [peak_range]*len(data_flattened)))

    area_image = np.array(areas).reshape(length, width)
    
    return area_image
