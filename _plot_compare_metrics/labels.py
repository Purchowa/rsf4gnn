from typing import Any, Dict, List, Optional, Tuple


def build_run_label(log: Dict[str, Any], file_path: str) -> str:
    model = str(log.get("model", "unknown_model"))
    data = str(log.get("data", "unknown_data"))
    meta = log.get("meta", {}) if isinstance(log.get("meta", {}), dict) else {}

    sampling_enabled = meta.get("sampling_enabled", None)
    reordering_applied = meta.get("reordering_applied", None)
    reordering_method = meta.get("reordering_method", None)

    parts = [model, data]
    if sampling_enabled is True:
        parts.append("sampling=on")
    elif sampling_enabled is False:
        parts.append("sampling=off")

    if reordering_applied is True:
        if isinstance(reordering_method, str) and reordering_method and reordering_method != "none":
            parts.append("reordering=" + reordering_method)
        else:
            parts.append("reordering=on")
    elif reordering_applied is False:
        parts.append("reordering=off")

    return " | ".join(parts)


def display_metric_name(metric: str) -> str:
    if metric.endswith("_sec"):
        return metric[:-4] + " [s]"
    return metric


def compress_labels_for_axis(
    labels: List[str],
    run_indices: Optional[List[int]] = None,
) -> Tuple[str, List[str]]:
    split_labels = [[part.strip() for part in label.split("|")] for label in labels]
    if not split_labels:
        return "", []

    common_parts: List[str] = []
    for part in split_labels[0]:
        if all(part in parts for parts in split_labels[1:]):
            common_parts.append(part)

    unique_labels: List[str] = []
    for i, parts in enumerate(split_labels):
        unique_parts = [part for part in parts if part not in common_parts]
        if unique_parts:
            unique_labels.append(" | ".join(unique_parts))
            continue

        if run_indices is not None and i < len(run_indices):
            unique_labels.append("run=" + str(run_indices[i]))
        else:
            unique_labels.append("run=" + str(i))

    common_context = " | ".join(common_parts)
    return common_context, unique_labels
