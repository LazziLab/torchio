import nibabel as nib
import numpy as np

from ....data.lazy_subject import Lazy_Subject
from .bounds_transform import BoundsTransform
from .bounds_transform import TypeBounds

# lazy cropping passes a view instead of cloning (numpy and zarr don't have .clone())
class Lazy_Crop(BoundsTransform):
    r"""Crop an image.

    Args:
        cropping: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values cropped from the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(- w_{ini} + W - w_{fin}) \times (- h_{ini} + H - h_{fin})
            \times (- d_{ini} + D - d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin}
            = d_{ini} = d_{fin} = n`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.
    """

    def __init__(self, cropping: TypeBounds, **kwargs):
        super().__init__(cropping, **kwargs)
        self.cropping = cropping
        self.args_names = ['cropping']

    def apply_transform(self, sample) -> Lazy_Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        index_ini = low
        index_fin = np.array(sample.spatial_shape) - high
        for image in self.get_images(sample):
            new_origin = nib.affines.apply_affine(image.affine, index_ini)
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            i0, j0, k0 = index_ini
            i1, j1, k1 = index_fin
            image.set_data(image.data[:, i0:i1, j0:j1, k0:k1])  #removed clone()
            image.affine = new_affine
        return sample

    def inverse(self):
        from .pad import Pad

        return Pad(self.cropping)
