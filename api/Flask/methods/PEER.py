import sys
sys.path.append('/media/ramancloud/api/')
from PEER import peer


def peer_process(spectrum, wavenumbers, loops=6, half_k_threshold=3):

    denoised_spectrum = peer(spectrum, loops, half_k_threshold)
    original_wavenumbers = wavenumbers.copy()

    return denoised_spectrum, original_wavenumbers