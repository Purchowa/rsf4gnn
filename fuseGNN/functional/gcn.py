import torch
import fgnn_gcn

# fused get edge weight function for GAR model

class Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree):
        self_edge_weight = 1./degree
        return self_edge_weight
    @staticmethod
    def backward(ctx, gred_self_edge_weight):
        return None

class GCNGAREdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow):
        edge_weight, degree = fgnn_gcn.gcn_gar_edge_weight(src_index, tar_ptr, tar_index, num_nodes, edge_weight,
                                                             flow=='target_to_source')
        return edge_weight, degree
    
    @staticmethod
    def backward(ctx, grad_edge_weight, grad_degree):
        return None, None, None, None, None, None

def gcn_gar_edge_weight(src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow,
                        sampling_debug_enabled=False, sampling_debug_fail_fast=False,
                        sampling_debug_split='unknown', sampling_debug_batch=-1, sampling_debug_layer=-1):
    if sampling_debug_enabled:
        print(
            '[SamplingDebug][GAREdgeWeight] split={} batch={} layer={} num_nodes={} '
            'src_edges={} tar_ptr_rows={} tar_edges={}'.format(
                sampling_debug_split,
                sampling_debug_batch,
                sampling_debug_layer,
                int(num_nodes),
                int(src_index.size(0)),
                int(tar_ptr.size(0)),
                int(tar_index.size(0)),
            )
        )

    edge_weight, degree = GCNGAREdgeWeight.apply(src_index, tar_ptr, tar_index, num_nodes, edge_weight, flow)

    if sampling_debug_enabled:
        degree_min = float(degree.min().item()) if degree.numel() > 0 else 0.0
        degree_max = float(degree.max().item()) if degree.numel() > 0 else 0.0
        zero_count = int((degree == 0).sum().item()) if degree.numel() > 0 else 0
        inf_count = int(torch.isinf(degree).sum().item()) if degree.numel() > 0 else 0
        nan_count = int(torch.isnan(degree).sum().item()) if degree.numel() > 0 else 0
        print(
            '[SamplingDebug][GAREdgeWeight] split={} batch={} layer={} '
            'degree_min={} degree_max={} zero_count={} inf_count={} nan_count={}'.format(
                sampling_debug_split,
                sampling_debug_batch,
                sampling_debug_layer,
                degree_min,
                degree_max,
                zero_count,
                inf_count,
                nan_count,
            )
        )

    if sampling_debug_fail_fast:
        if degree.numel() > 0 and (torch.isnan(degree).any() or torch.isinf(degree).any()):
            raise RuntimeError(
                'GAR degree tensor contains NaN/Inf before reciprocal: split={} batch={} layer={}'.format(
                    sampling_debug_split, sampling_debug_batch, sampling_debug_layer
                )
            )

    self_edge_weight = Inv.apply(degree)
    return edge_weight, self_edge_weight
    
# fused get edge weight function for GAS model

class GCNGASEdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_index, tar_index, num_nodes, edge_weight, flow):
        weight_to_cache, degree_to_cache = fgnn_gcn.gcn_gas_edge_weight(src_index, tar_index, num_nodes, edge_weight,
                                                                          flow=='target_to_source')
        return weight_to_cache, degree_to_cache
    
    @staticmethod
    def backward(ctx, grad_edge_weight, grad_degree):
        return None, None, None, None, None, None
    
def gcn_gas_edge_weight(src_index, tar_index, num_nodes, edge_weight, flow,
                        sampling_debug_enabled=False, sampling_debug_fail_fast=False,
                        sampling_debug_split='unknown', sampling_debug_batch=-1, sampling_debug_layer=-1):
    if sampling_debug_enabled:
        print(
            '[SamplingDebug][GASEdgeWeight] split={} batch={} layer={} num_nodes={} edges={}'.format(
                sampling_debug_split,
                sampling_debug_batch,
                sampling_debug_layer,
                int(num_nodes),
                int(src_index.size(0)),
            )
        )

    edge_weight, degree = GCNGASEdgeWeight.apply(src_index, tar_index, num_nodes, edge_weight, flow)

    if sampling_debug_enabled:
        degree_min = float(degree.min().item()) if degree.numel() > 0 else 0.0
        degree_max = float(degree.max().item()) if degree.numel() > 0 else 0.0
        zero_count = int((degree == 0).sum().item()) if degree.numel() > 0 else 0
        inf_count = int(torch.isinf(degree).sum().item()) if degree.numel() > 0 else 0
        nan_count = int(torch.isnan(degree).sum().item()) if degree.numel() > 0 else 0
        print(
            '[SamplingDebug][GASEdgeWeight] split={} batch={} layer={} '
            'degree_min={} degree_max={} zero_count={} inf_count={} nan_count={}'.format(
                sampling_debug_split,
                sampling_debug_batch,
                sampling_debug_layer,
                degree_min,
                degree_max,
                zero_count,
                inf_count,
                nan_count,
            )
        )

    if sampling_debug_fail_fast:
        if degree.numel() > 0 and (torch.isnan(degree).any() or torch.isinf(degree).any()):
            raise RuntimeError(
                'GAS degree tensor contains NaN/Inf before reciprocal: split={} batch={} layer={}'.format(
                    sampling_debug_split, sampling_debug_batch, sampling_debug_layer
                )
            )

    self_edge_weight = Inv.apply(degree)
    return edge_weight, self_edge_weight
