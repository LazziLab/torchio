from typing import Generator
from typing import Optional

import torch

from ...data.lazy_subject import Lazy_Subject
from .lazy_sampler import Lazy_RandomSampler


class Lazy_UniformSampler(Lazy_RandomSampler):
    """Randomly extract patches from a volume with uniform probability.

    Args:
        patch_size: See :class:`~torchio.data.Lazy_PatchSampler`.
    """

    def get_probability_map(self, subject: Lazy_Subject) -> torch.Tensor:
        return torch.ones(1, *subject.spatial_shape)

    def _generate_patches(
        self,
        subject: Lazy_Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Lazy_Subject, None, None]:
        valid_range = subject.spatial_shape - self.patch_size
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            i, j, k = tuple(int(torch.randint(x + 1, (1,)).item()) for x in valid_range)
            index_ini = i, j, k
            yield self.extract_patch(subject, index_ini)
            if num_patches is not None:
                patches_left -= 1
