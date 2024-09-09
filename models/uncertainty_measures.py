""" Taken from https://github.com/cai4cai/ACE-DLIRIS/blob/main/ace_dliris/losses/hardl1ace.py
https://arxiv.org/pdf/2403.06759 Barfoot et al.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction
from monai.losses import DiceLoss


__all__ = [
    "hard_binned_calibration",
    "HardL1ACELoss",
    "HardL1ACEandCELoss",
    "HardL1ACEandDiceLoss",
    "HardL1ACEandDiceCELoss",
]


def hard_binned_calibration(input, target, num_bins=20, right=False):
    """
    Compute the calibration bins for the given data. This function calculates the mean predictions,
    mean ground truths, and bin counts for each bin in a hard binning calibration approach.

    The function operates on input and target tensors with batch and channel dimensions,
    handling each batch and channel separately. For bins that do not contain any elements,
    the mean predicted values and mean ground truth values are set to NaN.


    Args:
        input (torch.Tensor): Input tensor with shape [batch, channel, spatial], where spatial
            can be any number of dimensions. The input tensor represents predicted values or probabilities.
        target (torch.Tensor): Target tensor with the same shape as input. It represents ground truth values.
        num_bins (int, optional): The number of bins to use for calibration. Defaults to 20.
        right (bool, optional): If False (default), the bins include the left boundary and exclude the right boundary.
            If True, the bins exclude the left boundary and include the right boundary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - mean_p_per_bin (torch.Tensor): Tensor of shape [batch_size, num_channels, num_bins] containing
              the mean predicted values in each bin.
            - mean_gt_per_bin (torch.Tensor): Tensor of shape [batch_size, num_channels, num_bins] containing
              the mean ground truth values in each bin.
            - bin_counts (torch.Tensor): Tensor of shape [batch_size, num_channels, num_bins] containing
              the count of elements in each bin.

    Raises:
        ValueError: If the input and target shapes do not match or if the input is not three-dimensional.

    Note:
        This function currently uses nested for loops over batch and channel dimensions
        for binning operations. Future improvements may include vectorizing these operations
        for enhanced performance.
    """
    if input.shape != target.shape:
        raise ValueError(
            f"Input and target should have the same shapes, got {input.shape} and {target.shape}."
        )
    if input.dim() < 3:
        raise ValueError(
            f"Input should be at least a three-dimensional tensor, got {input.dim()} dimensions."
        )

    batch_size, num_channels = input.shape[:2]
    boundaries = torch.linspace(
        start=0.0,
        end=1.0 - torch.finfo(torch.float32).eps,
        steps=num_bins + 1,
        device=input.device,
    )

    mean_p_per_bin = torch.zeros(
        batch_size, num_channels, num_bins, device=input.device
    )
    mean_gt_per_bin = torch.zeros_like(mean_p_per_bin)
    bin_counts = torch.zeros_like(mean_p_per_bin)

    input = input.flatten(start_dim=2).float()
    target = target.flatten(start_dim=2).float()

    for b in range(batch_size):
        for c in range(num_channels):
            bin_idx = torch.bucketize(input[b, c, :], boundaries[1:], right=right)
            bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)
            bin_counts[b, c, :] = torch.zeros_like(boundaries[1:]).scatter_add(
                0, bin_idx, torch.ones_like(input[b, c, :])
            )

            mean_p_per_bin[b, c, :] = torch.empty_like(boundaries[1:]).scatter_reduce(
                0, bin_idx, input[b, c, :], reduce="mean", include_self=False
            )
            mean_gt_per_bin[b, c, :] = torch.empty_like(boundaries[1:]).scatter_reduce(
                0, bin_idx, target[b, c, :].float(), reduce="mean", include_self=False
            )

    # Remove nonsense bins:
    mean_p_per_bin[bin_counts == 0] = torch.nan
    mean_gt_per_bin[bin_counts == 0] = torch.nan

    return mean_p_per_bin, mean_gt_per_bin, bin_counts


class HardL1ACELoss(_Loss):
    """
    Hard Binned L1 Average Calibration Error (ACE) loss.

    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        right: bool = False,
    ) -> None:
        """
        Args:
            num_bins: the number of bins to use for the binned L1 ACE loss calculation. Defaults to 20.
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            right: If False (default), the bins include the left boundary and exclude the right boundary.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.num_bins = num_bins
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.weight = weight
        self.right = right
        self.register_buffer("class_weight", torch.ones(1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.....
        """
        # TODO: may need error handling if input is not in the range [0, 1] - as this will throw an error in bucketize

        if self.sigmoid:
            input = torch.sigmoid(input)

        # batch_size = input.shape[0]
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        # CALCULATE ACE LOSS:
        mean_p_per_bin, mean_gt_per_bin, bin_counts = hard_binned_calibration(
            input, target, num_bins=self.num_bins, right=self.right
        )
        f = torch.nanmean(torch.abs(mean_p_per_bin - mean_gt_per_bin), dim=-1)

        if self.weight is not None and target.shape[1] != 1:
            # make sure the lengths of weights are equal to the number of classes
            num_of_classes = target.shape[1]
            if isinstance(self.weight, (float, int)):
                self.class_weight = torch.as_tensor([self.weight] * num_of_classes)
            else:
                self.class_weight = torch.as_tensor(self.weight)
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError(
                    "the value/values of the `weight` should be no less than 0."
                )
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return f  # L1 ACE loss


class HardL1ACEandCELoss(_Loss):
    """
    A class that combines L1 ACE Loss and CrossEntropyLoss with specified weights.
    """

    def __init__(
        self,
        ace_weight=0.5,
        ce_weight=0.5,
        to_onehot_y=False,
        ace_params=None,
        ce_params=None,
    ):
        """
        Initializes the HardL1ACEandCELoss class.

        Args:
            ace_weight (float): Weight for the HardL1ACELoss component.
            ce_weight (float): Weight for the CrossEntropyLoss component.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `pred` (``pred.shape[1]``). Defaults to False.
            ace_params (dict, optional): Parameters for the HardL1ACELoss.
            ce_params (dict, optional): Parameters for the CrossEntropyLoss.
        """
        super().__init__()
        self.ace_weight = ace_weight
        self.ce_weight = ce_weight
        self.to_onehot_y = to_onehot_y
        self.ace_loss = HardL1ACELoss(**(ace_params if ace_params is not None else {}))
        self.ce_loss = nn.CrossEntropyLoss(
            **(ce_params if ce_params is not None else {})
        )

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the weighted sum of L1 ACE and CrossEntropy losses.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The weighted sum of L1 ACE and CrossEntropy losses.
        """
        # TODO: need to think about how reductions are handles for the two losses when combining
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        ace_loss_val = self.ace_loss(y_pred, y_true)
        ce_loss_val = self.ce_loss(y_pred, y_true)
        return self.ace_weight * ace_loss_val + self.ce_weight * ce_loss_val


class HardL1ACEandDiceLoss(_Loss):
    """
    A class that combines L1 ACE Loss and DiceLoss with specified weights.
    """

    def __init__(
        self,
        ace_weight=0.5,
        dice_weight=0.5,
        to_onehot_y=False,
        ace_params=None,
        dice_params=None,
    ):
        """
        Initializes the HardL1ACEandCELoss class.

        Args:
            ace_weight (float): Weight for the HardL1ACELoss component.
            dice_weight (float): Weight for the DiceLoss component.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `pred` (``pred.shape[1]``). Defaults to False.
            ace_params (dict, optional): Parameters for the HardL1ACELoss.
            dice_params (dict, optional): Parameters for the DiceLoss.
        """
        super().__init__()
        self.ace_weight = ace_weight
        self.dice_weight = dice_weight
        self.to_onehot_y = to_onehot_y
        self.ace_loss = HardL1ACELoss(**(ace_params if ace_params is not None else {}))
        self.dice_loss = DiceLoss(**(dice_params if dice_params is not None else {}))

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the weighted sum of L1 ACE and Dice losses.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The weighted sum of L1 ACE and Dice losses.
        """
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        ace_loss_val = self.ace_loss(y_pred, y_true)
        dice_loss_val = self.dice_loss(y_pred, y_true)
        return self.ace_weight * ace_loss_val + self.dice_weight * dice_loss_val


class HardL1ACEandDiceCELoss(_Loss):
    """
    A class that combines L1 ACE Loss, Dice Loss, and CrossEntropyLoss with specified weights.
    """

    def __init__(
        self,
        ace_weight=0.33,
        ce_weight=0.33,
        dice_weight=0.33,
        to_onehot_y=False,
        ace_params=None,
        dice_params=None,
        ce_params=None,
    ):
        """
        Initializes the HardL1ACEandDiceCELoss class.

        Args:
            ace_weight (float): Weight for the HardL1ACELoss component.
            dice_weight (float): Weight for the DiceLoss component.
            ce_weight (float): Weight for the CrossEntropyLoss component.
            to_onehot_y (bool): Whether to convert the `target` into the one-hot format.
            ace_params (dict, optional): Parameters for the HardL1ACELoss.
            dice_params (dict, optional): Parameters for the DiceLoss.
            ce_params (dict, optional): Parameters for the CrossEntropyLoss.
        """
        super().__init__()
        self.ace_weight = ace_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.to_onehot_y = to_onehot_y

        self.ace_loss = HardL1ACELoss(**(ace_params if ace_params is not None else {}))
        self.dice_loss = DiceLoss(**(dice_params if dice_params is not None else {}))
        self.ce_loss = nn.CrossEntropyLoss(
            **(ce_params if ce_params is not None else {})
        )

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the weighted sum of L1 ACE, Dice, and CrossEntropy losses.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The weighted sum of L1 ACE, Dice, and CrossEntropy losses.
        """
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        ace_loss_val = self.ace_loss(y_pred, y_true)
        dice_loss_val = self.dice_loss(y_pred, y_true)
        ce_loss_val = self.ce_loss(y_pred, y_true)
        return (
            self.ace_weight * ace_loss_val
            + self.dice_weight * dice_loss_val
            + self.ce_weight * ce_loss_val
        )
    
""" Code taken from https://github.com/RagMeh11/QU-BraTS
from paper QU-BraTS: MICCAI BraTS 2020 Challenge on Quantifying Uncertainty in Brain Tumor 
Segmentation - Analysis of Ranking Scores and Benchmarking Results
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10060060/"""
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import nibabel as nib

#############################################3
### Pretty plots
import matplotlib.pyplot as plt

from matplotlib import cycler

plt.style.use("default")
plt.rcParams.update(
  {"lines.linewidth": 1.5,
   "axes.grid": True,
   "grid.linestyle": ":",
   "axes.grid.axis": "both",
   "axes.prop_cycle": cycler('color',
                             ['0071bc', 'd85218', 'ecb01f',
                              '7d2e8d', '76ab2f', '4cbded', 'a1132e']),
   "xtick.top": True,
   "xtick.minor.size": 0,
   "xtick.direction": "in",
   "xtick.minor.visible": True,
   "ytick.right": True,
   "ytick.minor.size": 0,
   "ytick.direction": "in",
   "ytick.minor.visible": True,
   "legend.framealpha": 1.0,
   "legend.edgecolor": "black",
   "legend.fancybox": False,
   "figure.figsize": (2.5, 2.5),
   "figure.autolayout": False,
   "savefig.dpi": 300,
   "savefig.format": "png",
   "savefig.bbox": "tight",
   "savefig.pad_inches": 0.01,
   "savefig.transparent": False
  }
)

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

###############################################

EPS = np.finfo(np.float32).eps

def dice_metric(ground_truth, predictions):
    """

    Returns Dice coefficient for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target, 
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].

    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.

    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")

    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
      dice = 1.0
    else:
      dice = (2. * intersection) / (union)

    return dice


def ftp_ratio_metric(ground_truth, 
                     predictions, 
                     unc_mask,
                     brain_mask):
    """

    Returns Filtered True Positive Ratio for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                    with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentatation predictions,
                    with shape [W, H, D].
      unc_mask:     `numpy.ndarray`, uncertainty binary mask, where uncertain voxels has value 0 
                    and certain voxels has value 1, with shape [W, H, D].
      brain_mask:   `numpy.ndarray`, brain binary mask, where background voxels has value 0 
                    and forground voxels has value 1, with shape [W, H, D].

    Returns:
      Filtered true positive ratio (`float` in [0.0, 1.0]).

    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")
    unc_mask = unc_mask.astype("float32")
    brain_mask = brain_mask.astype("float32")

    # Get filtered Filtered TP ratio (We use EPS for numeric stability)
    TP = (predictions * ground_truth) * brain_mask
    tp_before_filtering = TP.sum()  # TP before filtering
    tp_after_filtering = (TP * unc_mask).sum()  # TP after filtering

    ftp_ratio = (tp_before_filtering - tp_after_filtering) / (tp_before_filtering + EPS)

    return ftp_ratio


def ftn_ratio_metric(ground_truth, 
                     predictions, 
                     unc_mask,
                     brain_mask):
    """

    Returns Filtered True Negative Ratio for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                    with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentatation predictions,
                    with shape [W, H, D].
      unc_mask:     `numpy.ndarray`, uncertainty binary mask, where uncertain voxels has value 0 
                    and certain voxels has value 1, with shape [W, H, D].
      brain_mask:   `numpy.ndarray`, brain binary mask, where background voxels has value 0 
                    and forground voxels has value 1, with shape [W, H, D].

    Returns:
      Filtered true negative ratio (`float` in [0.0, 1.0]).

    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")
    unc_mask = unc_mask.astype("float32")
    brain_mask = brain_mask.astype("float32")

    # Get filtered Filtered TN ratio (We use EPS for numeric stability)
    TN = ((1-predictions) * (1-ground_truth)) * brain_mask
    tn_before_filtering = TN.sum() # TN before filtering
    tn_after_filtering = (TN * unc_mask).sum() # TN after filtering

    ftn_ratio = (tn_before_filtering - tn_after_filtering) / (tn_before_filtering + EPS)

    return ftn_ratio


def make(ground_truth, 
         predictions, 
         uncertainties,
         brain_mask, 
         thresholds):

    """
    Performs evaluation for a binary segmentation task.

    Args:
      ground_truth:  `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [W, H, D].
      predictions:   `numpy.ndarray`, binary segmentatation predictions,
                     with shape [W, H, D].
      uncertainties: `numpy.ndarray`, uncertainties for `predictions`, 
                     with values in [0, 100] and shape [W, H, D].
      brain_mask:    `numpy.ndarray`, binary brain mask, 
                     with shape [W, H, D].
      thresholds:    `numpy.ndarray`, the cut-off values for `uncertainties`,
                     with shape [K].

    Returns:
      dice: `numpy.ndarray`, the dice for the different uncertainty
        `thresholds`, with shape [K].
      ftp_ratio: `numpy.ndarray`, the FTP ratio for the different uncertainty
        `thresholds`, with shape [K].
      ftn_ratio: `numpy.ndarray`, the FTN ratio for the different uncertainty
        `thresholds`, with shape [K].
    """

    dice = list()
    ftp_ratio = list()
    ftn_ratio = list()

    # Iterate through different uncertainty thresholds
    for th in thresholds:

        # Convert uncertainty to binary mask according to uncertainty threshold
        # voxels with uncertainty greater than threshold are considered uncertain
        # and voxels with uncertainty less than threshold are considered certain
        unc_mask = np.ones_like(uncertainties, dtype='float32')
        unc_mask[uncertainties > th] = 0.0

        # Multiply ground truth and predictions with unc_mask his helps in filtering out uncertain voxels
        # we calculate metric of interest (here, dice) only on unfiltered certain voxels
        ground_truth_filtered = ground_truth * unc_mask
        predictions_filtered = predictions * unc_mask

        # Calculate dice
        dsc_i = dice_metric(ground_truth_filtered, predictions_filtered)
        dice.append(dsc_i)

        # Calculate filtered true positive ratio
        ftp_ratio_i = ftp_ratio_metric(ground_truth, predictions, unc_mask, brain_mask)
        ftp_ratio.append(ftp_ratio_i)

        # Calculate filtered true negative ratio
        ftn_ratio_i = ftn_ratio_metric(ground_truth, predictions, unc_mask, brain_mask)
        ftn_ratio.append(ftn_ratio_i)

    return dice, ftp_ratio, ftn_ratio



def evaluate(ground_truth,
             segmentation,
             whole,
             core,
             enhance,
             brain_mask,
             output_file,
             num_points,
             return_auc=True,
             return_plot=True):
  
    """
    Evaluates a single sample from BraTS.

    Args:
        ground_truth: `str`, path to ground truth segmentation .
        segmentation: `str`, path to segmentation map.
        whole: `str`, path to uncertainty map for whole tumor.
        core: `str`, path to uncertainty map for core tumor.
        enhance: `str`, path to uncertainty map for enhance tumor.
        brain_mask: `str`, path to brain mask.
        output_file: `str`, path to output file to store statistics.
        num_points: `int`, number of uncertainty threshold points.
        return_auc: `bool`, if it is True it returns AUCs.
        return_plot: `bool`, if it is True it returns plots (Dice vs 1 - Unc_thresholds, FTP vs 1 - Unc_thresholds, FTN vs 1 - Unc_thresholds).

    Returns:
        The table (`pandas.DataFrame`) that summarizes the metrics.
    """

    # Define Uncertainty Threshold points
    _UNC_POINTs = np.arange(0.0, 100.0 + EPS, 100.0 / num_points).tolist()
    _UNC_POINTs.reverse()

    # Parse NIFTI files
    GT = nib.load(ground_truth).get_fdata()
    PRED = nib.load(segmentation).get_fdata()
    WT = nib.load(whole).get_fdata()
    TC = nib.load(core).get_fdata()
    ET = nib.load(enhance).get_fdata()
    BM = nib.load(brain_mask).get_fdata()


    # convert mask into binary.
    # useful when you don't have access to the mask, but generating it from T1 image
    # 0 intensity is considered background, anything else is forground
    # works well with BraTS
    BM[BM>0] = 1.0
    
    # Output container
    METRICS = dict()

    ########
    # Whole Tumour: take 1,2, and 4 label as foreground, 0 as background.

    # convert multi-Label GT and Pred to binary class
    GT_bin = np.zeros_like(GT)
    Pred_bin = np.zeros_like(PRED)

    GT_bin[GT > 0] = 1.0
    Pred_bin[PRED > 0] = 1.0

    METRICS["WT_DICE"], METRICS["WT_FTP_RATIO"], METRICS["WT_FTN_RATIO"] = make(GT_bin, Pred_bin, WT, BM, _UNC_POINTs)

    #######
    # Tumour Core: take 1 and 4 label as foreground, 0 and 2 as background.

    # convert multi-Label GT and Pred to binary class
    GT_bin = np.zeros_like(GT)
    Pred_bin = np.zeros_like(PRED)

    GT_bin[GT == 1] = 1.0
    GT_bin[GT == 4] = 1.0
    Pred_bin[PRED == 1] = 1.0
    Pred_bin[PRED == 4] = 1.0

    METRICS["TC_DICE"], METRICS["TC_FTP_RATIO"], METRICS["TC_FTN_RATIO"] = make(GT_bin, Pred_bin, TC, BM, _UNC_POINTs)

    ##########
    # Enhancing Tumour: take 4 label as foreground, 0, 1, and 2 as bacground.

    # convert multi-Label GT and Pred to binary class
    GT_bin = np.zeros_like(GT)
    Pred_bin = np.zeros_like(PRED)

    GT_bin[GT == 4] = 1.0
    Pred_bin[PRED == 4] = 1.0

    METRICS["ET_DICE"], METRICS["ET_FTP_RATIO"], METRICS["ET_FTN_RATIO"] = make(GT_bin, Pred_bin, ET, BM, _UNC_POINTs)


    ##########
    # save plot

    if return_plot:

        # create a plot for Dice vs 100 - Unc_Thres, FTP vs 100 - Unc_Thres, FTN vs 100 - Unc_Thres for all three tumour types: WT, TC, ET 
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12.0, 8.0), sharex=True)
        
        # loop through Different Metrics
        for j, met in enumerate(["DICE", "FTP_RATIO", "FTN_RATIO"]):

             # loop throug different Tumour Type
            for i, typ in enumerate(["WT", "TC", "ET"]): 

            	# plot 100 - _Unc_threshold on X-axis and Metric on Y-axis. Calculate AUC also
                axes[j,i].plot(100 - np.array(_UNC_POINTs), np.array(METRICS[typ+"_"+met]), 
               	               color=COLORS[i], 
               	               # marker='o', 
               	               label='AUC: {:.4f}'.format(auc(100 - np.array(_UNC_POINTs), np.array(METRICS[typ+"_"+met]))/100.0))

                # set ylabel for first column
                if i == 0:
                    axes[j, i].set(ylabel=met)

                # set title for first row
                if j == 0:
                    axes[j, i].set(title=typ) 
        
                # set xlabel for last row          
                if j == 2:
                    axes[j, i].set(xlabel="1 - Uncertainty Threshold")

                axes[j,i].set(ylim = (0.00,1.0001))
                axes[j,i].set(xlim = (0.00,100.0001))


        [ax.legend() for ax in axes.flatten()]

        fig.savefig(output_file+'.png', dpi=300, format="png", trasparent=True)


    ################
    # Print to CSV

    if not return_auc:
    
        # Returns <thresholds: [DICE_{type}, FTP_RATIO_{type}, FTN_RATIO_{type}]>
        METRICS["THRESHOLDS"] = _UNC_POINTs
        df = pd.DataFrame(METRICS).set_index("THRESHOLDS")
        df.to_csv(output_file+'.csv')
    
        return df
    
    else:
    
        # Returns <{type}: [DICE_AUC, FTP_RATIO_AUC, FTN_RATIO_AUC]>
        df = pd.DataFrame(index=["WT", "TC", "ET"],
                          columns=["DICE_AUC", "FTP_RATIO_AUC", "FTN_RATIO_AUC"],
                          dtype=float)
        
        for ttype in df.index:
            df.loc[ttype, "DICE_AUC"]      = auc(_UNC_POINTs, METRICS["{}_DICE".format(ttype)]) / 100.0
            df.loc[ttype, "FTP_RATIO_AUC"] = auc(_UNC_POINTs, METRICS["{}_FTP_RATIO".format(ttype)]) / 100.0
            df.loc[ttype, "FTN_RATIO_AUC"] = auc(_UNC_POINTs, METRICS["{}_FTN_RATIO".format(ttype)]) / 100.0
    
        df.index.names = ["TUMOR_TYPE"]
        df.to_csv(output_file+'.csv')
    
        return df