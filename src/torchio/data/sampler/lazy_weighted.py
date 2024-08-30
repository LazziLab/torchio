from typing import Generator
from typing import Optional, Union, Tuple, Literal

import numpy as np
import torch

from ...constants import MIN_FLOAT_32
from ...typing import TypeSpatialShape
from ..lazy_image import Lazy_Image
from ..lazy_subject import Lazy_Subject
from .lazy_sampler import Lazy_RandomSampler


class Lazy_WeightedSampler(Lazy_RandomSampler):
    r"""Randomly extract patches from a volume given a probability map.

    The probability of sampling a patch centered on a specific voxel is the
    value of that voxel in the probability map. The probabilities need not be
    normalized. For example, voxels can have values 0, 1 and 5. Voxels with
    value 0 will never be at the center of a patch. Voxels with value 5 will
    have 5 times more chance of being at the center of a patch that voxels
    with a value of 1.

    Args:
        patch_size: See :class:`~torchio.data.Lazy_PatchSampler`.
        probability_map: Name of the image in the input subject that will be
            used as a sampling probability map.

    Raises:
        RuntimeError: If the probability map is empty.

    Example:
        >>> import torchio as tio
        >>> subject = tio.Lazy_Subject(
        ...     t1=tio.ScalarImage('t1_mri.nii.gz'),
        ...     sampling_map=tio.Lazy_Image('sampling.nii.gz', type=tio.SAMPLING_MAP),
        ... )
        >>> patch_size = 64
        >>> sampler = tio.data.Lazy_WeightedSampler(patch_size, 'sampling_map')
        >>> for patch in sampler(subject):
        ...     print(patch[tio.LOCATION])

    .. note:: The index of the center of a patch with even size :math:`s` is
        arbitrarily set to :math:`s/2`. This is an implementation detail that
        will typically not make any difference in practice.

    .. note:: Values of the probability map near the border will be set to 0 as
        the center of the patch cannot be at the border (unless the patch has
        size 1 or 2 along that axis).
    """  # noqa: B950

    def __init__(
        self,
        patch_size: TypeSpatialShape,
        probability_map: Optional[str],
    ):
        super().__init__(patch_size)
        self.probability_map_name = probability_map
        self.cdf = None

    def _generate_patches(
        self,
        subject: Lazy_Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Lazy_Subject, None, None]:
        probability_map = self.get_probability_map(subject)
        if type(probability_map) is torch.Tensor:
            probability_map_array = self.process_probability_map(
                probability_map,
                subject,
            )
            cdf = self.get_cumulative_distribution_function(probability_map_array)

            patches_left = num_patches if num_patches is not None else True
            while patches_left:
                yield self.extract_patch(subject, probability_map_array, cdf)
                if num_patches is not None:
                    patches_left -= 1
        else:
            patches_left = num_patches if num_patches is not None else True
            
            ## Pre-process k-dimension once per image
            crop_ini = self.patch_size // 2
            crop_ii, crop_ji, crop_ki = crop_ini
            crop_fin = (self.patch_size - 1) // 2
            crop_if, crop_jf, crop_kf = crop_fin.tolist()
            # The call tolist() is very important. Using np.uint16 as negative
            # index will not work because e.g. -np.uint16(2) == 65534
            crop_if = -crop_if if crop_if>0 else None
            crop_jf = -crop_jf if crop_jf>0 else None
            crop_kf = -crop_kf if crop_kf>0 else None
            crop_ini = (crop_ii, crop_ji, crop_ki)
            crop_fin = (crop_if, crop_jf, crop_kf)
            # slice map to exclude edges where no patch should be centered
            probability_map_edgeSliced = probability_map.lazy_slice[:, crop_ii:crop_if, crop_ji:crop_jf, crop_ki:crop_kf]
            # sum along all but k-dimension
            sum_crop_to_k = np.sum(probability_map_edgeSliced, axis=tuple([dim for dim in range(probability_map.ndim) if dim !=3]))
            # get total for normalization
            total = np.sum(sum_crop_to_k, dtype=np.float64)
            normalized_crop_to_k = sum_crop_to_k/total
            
            while patches_left:
                yield self.lazy_extract_patch(subject, probability_map, normalized_crop_to_k, crop_ini, crop_fin)
                if num_patches is not None:
                    patches_left -= 1

    def get_probability_map_image(self, subject: Lazy_Subject) -> Lazy_Image:
        assert self.probability_map_name is not None
        if self.probability_map_name in subject:
            return subject[self.probability_map_name]
        else:
            message = (
                f'Lazy_Image "{self.probability_map_name}" not found in subject: {subject}'
            )
            raise KeyError(message)

    def get_probability_map(self, subject: Lazy_Subject) -> torch.Tensor:
        data = self.get_probability_map_image(subject).data
        # if torch.any(data < 0):
        #     message = (
        #         'Negative values found'
        #         f' in probability map "{self.probability_map_name}"'
        #     )
        #     raise ValueError(message)
        return data

    def process_probability_map(
        self,
        probability_map: torch.Tensor,
        subject: Lazy_Subject,
    ) -> np.ndarray:
        # Using float32 can create cdf with maximum very far from 1, e.g. 0.92!
        data = probability_map[0].numpy().astype(np.float64)
        assert data.ndim == 3
        self.clear_probability_borders(data, self.patch_size)
        total = data.sum()
        if total == 0:
            half_patch_size = tuple(n // 2 for n in self.patch_size)
            message = (
                'Empty probability map found:'
                f' {self.get_probability_map_image(subject).path}'
                '\nVoxels with positive probability might be near the image'
                ' border.\nIf you suspect that this is the case, try adding a'
                ' padding transform\nwith half the patch size:'
                f' torchio.Pad({half_patch_size})'
            )
            raise RuntimeError(message)
        data /= total  # normalize probabilities
        return data

    @staticmethod
    def clear_probability_borders(
        probability_map: np.ndarray,
        patch_size: np.ndarray,
    ) -> None:
        # Set probability to 0 on voxels that wouldn't possibly be sampled
        # given the current patch size
        # We will arbitrarily define the center of an array with even length
        # using the // Python operator
        # For example, the center of an array (3, 4) will be on (1, 2)
        #
        #   Patch         center
        #  . . . .        . . . .
        #  . . . .   ->   . . x .
        #  . . . .        . . . .
        #
        #
        #    Prob. map      After preprocessing
        #
        #  x x x x x x x       . . . . . . .
        #  x x x x x x x       . . x x x x .
        #  x x x x x x x  -->  . . x x x x .
        #  x x x x x x x  -->  . . x x x x .
        #  x x x x x x x       . . x x x x .
        #  x x x x x x x       . . . . . . .
        #
        # The dots represent removed probabilities, x mark possible locations
        crop_ini = patch_size // 2
        crop_fin = (patch_size - 1) // 2
        crop_i, crop_j, crop_k = crop_ini
        probability_map[:crop_i, :, :] = 0
        probability_map[:, :crop_j, :] = 0
        probability_map[:, :, :crop_k] = 0

        # The call tolist() is very important. Using np.uint16 as negative
        # index will not work because e.g. -np.uint16(2) == 65534
        crop_i, crop_j, crop_k = crop_fin.tolist()
        if crop_i:
            probability_map[-crop_i:, :, :] = 0
        if crop_j:
            probability_map[:, -crop_j:, :] = 0
        if crop_k:
            probability_map[:, :, -crop_k:] = 0

    @staticmethod
    def get_cumulative_distribution_function(
        probability_map: np.ndarray,
    ) -> np.ndarray:
        """Return the cumulative distribution function of a probability map."""
        flat_map = probability_map.flatten()
        flat_map_normalized = flat_map / flat_map.sum()
        cdf = np.cumsum(flat_map_normalized)
        return cdf
    
    def lazy_extract_patch(
        self,
        subject: Lazy_Subject,
        probability_map, #: lazy_ops.lazy_loading.DatasetViewzarr, #left commented due to monkey patching
        normalized_crop_to_k: np.ndarray,
        crop_ini: Tuple[int, int, int],
        crop_fin: Tuple[Union[int,Literal[None]], Union[int,Literal[None]], Union[int,Literal[None]]],
    ) -> Lazy_Subject:
        si, sj, sk = self.patch_size
        crop_ii, crop_ji, crop_ki = crop_ini
        crop_if, crop_jf, crop_kf = crop_fin
        rng = np.random.default_rng()
        ndim = probability_map.ndim
        
        """ Randomly select k index """
        # # slice map to exclude edges where no patch should be centered
        # probability_map_edgeSliced = probability_map.lazy_slice[:, crop_ii:crop_if, crop_ji:crop_jf, crop_ki:crop_kf]
        # # sum along all but k-dimension
        # sum_crop_to_k = np.sum(probability_map_edgeSliced, axis=tuple([dim for dim in range(ndim) if dim !=3]))
        # # get total for normalization
        # total = np.sum(sum_crop_to_k, dtype=np.float64)
        # normalized_crop_to_k = sum_crop_to_k/total
        # randomly select k-index from cropped map and convert from numpy array to int
        # cropping offset doesn't need correcting, as this automatically puts indices at corner of patch
        k = ( rng.choice(normalized_crop_to_k.size, size=1, p=normalized_crop_to_k) )[0]
        
        """ Randomly select j index """
        # slice map to exclude edges where no patch should be centered
        # also slice to already set k-patch-position (but keep dimension)
        probability_map_edgeSliced = probability_map.lazy_slice[:, crop_ii:crop_if, crop_ji:crop_jf, k:k+sk]
        # sum along all but j-dimension
        sum_crop_to_j = np.sum(probability_map_edgeSliced, axis=tuple([dim for dim in range(ndim) if dim !=2]), dtype=np.float64)
        # get total for normalization
        total = np.sum(sum_crop_to_j, dtype=np.float64)
        # randomly select j-index from cropped map and convert from numpy array to int
        # cropping offset doesn't need correcting, as this automatically puts indices at corner of patch
        j = ( rng.choice(sum_crop_to_j.size, size=1, p=sum_crop_to_j/total) )[0]
        
        """ Randomly select i index """
        # slice map to exclude edges where no patch should be centered
        # also slice to already set k&j-patch-positions (but keep dimensions)
        probability_map_edgeSliced = probability_map.lazy_slice[:, crop_ii:crop_if, j:j+sj, k:k+sk]
        # sum along all but i-dimension
        sum_crop_to_i = np.sum(probability_map_edgeSliced, axis=tuple([dim for dim in range(ndim) if dim !=1]), dtype=np.float64)
        # get total for normalization
        total = np.sum(sum_crop_to_i, dtype=np.float64)
        # randomly select i-index from cropped map and convert from numpy array to int
        # cropping offset doesn't need correcting, as this automatically puts indices at corner of patch
        i = ( rng.choice(sum_crop_to_i.size, size=1, p=sum_crop_to_i/total) )[0]
        
        # i, j, k = self.get_random_index_ini(probability_map, cdf)
        patch_size = si, sj, sk
        index_ini = i, j, k
        cropped_subject = self.crop(
            subject,
            index_ini,
            patch_size,
        )
        return cropped_subject
    
    def extract_patch(  # type: ignore[override]
        self,
        subject: Lazy_Subject,
        probability_map: np.ndarray,
        cdf: np.ndarray,
    ) -> Lazy_Subject:
        i, j, k = self.get_random_index_ini(probability_map, cdf)
        index_ini = i, j, k
        si, sj, sk = self.patch_size
        patch_size = si, sj, sk
        cropped_subject = self.crop(
            subject,
            index_ini,
            patch_size,
        )
        return cropped_subject

    def get_random_index_ini(
        self,
        probability_map: np.ndarray,
        cdf: np.ndarray,
    ) -> np.ndarray:
        center = self.sample_probability_map(probability_map, cdf)
        assert np.all(center >= 0)
        # See self.clear_probability_borders
        index_ini = center - self.patch_size // 2
        assert np.all(index_ini >= 0)
        return index_ini

    @classmethod
    def sample_probability_map(
        cls,
        probability_map: np.ndarray,
        cdf: np.ndarray,
    ) -> np.ndarray:
        """Inverse transform sampling.

        Example:
            >>> probability_map = np.array(
            ...    ((0,0,1,1,5,2,1,1,0),
            ...     (2,2,2,2,2,2,2,2,2)))
            >>> probability_map
            array([[0, 0, 1, 1, 5, 2, 1, 1, 0],
                   [2, 2, 2, 2, 2, 2, 2, 2, 2]])
            >>> histogram = np.zeros_like(probability_map)
            >>> for _ in range(100000):
            ...     histogram[Lazy_WeightedSampler.sample_probability_map(probability_map, cdf)] += 1  # doctest:+SKIP
            ...
            >>> histogram  # doctest:+SKIP
            array([[    0,     0,  3479,  3478, 17121,  7023,  3355,  3378,     0],
                   [ 6808,  6804,  6942,  6809,  6946,  6988,  7002,  6826,  7041]])
        """  # noqa: B950
        # Get first value larger than random number ensuring the random number
        # is not exactly 0 (see https://github.com/fepegar/torchio/issues/510)
        random_number = max(MIN_FLOAT_32, torch.rand(1).item()) * cdf[-1]

        random_location_index = np.searchsorted(cdf, random_number)

        center = np.unravel_index(
            random_location_index,
            probability_map.shape,
        )

        probability = probability_map[center]
        if probability <= 0:
            message = (
                'Error retrieving probability in weighted sampler.'
                ' Please report this issue at'
                ' https://github.com/fepegar/torchio/issues/new?labels=bug&template=bug_report.md'  # noqa: B950
            )
            raise RuntimeError(message)

        return np.array(center)
