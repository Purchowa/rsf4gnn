"""
Metrics computation utilities for model evaluation.
"""
import torch


def safe_div(num, den):
    """
    Safe division that returns 0.0 if denominator is 0.
    
    Args:
        num: numerator
        den: denominator
        
    Returns:
        float: result of num/den or 0.0 if den is 0
    """
    if den == 0:
        return 0.0
    return float(num) / float(den)


def compute_acc_micro_macro_f1(logits, labels, mask, num_classes):
    """
    Compute accuracy, micro-averaged F1 score, and macro-averaged F1 score.
    
    Args:
        logits (torch.Tensor): Model output logits [N, num_classes]
        labels (torch.Tensor): Ground truth labels [N]
        mask (torch.Tensor): Boolean mask for samples to evaluate [N]
        num_classes (int): Number of classes
        
    Returns:
        tuple: (accuracy, micro_f1, macro_f1)
    """
    if int(mask.sum().item()) == 0:
        return 0.0, 0.0, 0.0

    pred = logits[mask].max(1)[1]
    true = labels[mask]

    acc = safe_div(pred.eq(true).sum().item(), int(mask.sum().item()))
    micro_f1 = acc

    macro_f1_sum = 0.0
    for cls in range(num_classes):
        pred_pos = pred == cls
        true_pos = true == cls
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()
        denom = (2 * tp) + fp + fn
        class_f1 = safe_div(2 * tp, denom) if denom > 0 else 0.0
        macro_f1_sum += class_f1
    macro_f1 = macro_f1_sum / float(num_classes)

    return acc, micro_f1, macro_f1
