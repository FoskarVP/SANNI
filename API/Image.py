import numpy as np
from pyts.image import RecurrencePlot
from scipy.ndimage.interpolation import zoom


def subsequent_to_image(subsequent: np.ndarray) -> np.ndarray:
    rp = RecurrencePlot(threshold='point')

    buf = zoom(rp.transform(np.array(subsequent).reshape(1, -1))[0], 0.25)

    return buf
