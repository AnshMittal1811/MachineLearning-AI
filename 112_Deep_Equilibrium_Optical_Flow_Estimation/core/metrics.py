import torch

MAX_FLOW = 400

@torch.no_grad()
def compute_epe(flow_pred, flow_gt, valid, max_flow=MAX_FLOW):
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    epe = torch.masked_fill(epe, ~valid, 0)
    
    return epe


@torch.no_grad()
def process_metrics(epe, rel_error, abs_error, **kwargs):
    epe = epe.flatten(1)
    metrics = torch.stack(
        [
            epe.mean(dim=1), 
            (epe < 1).float().mean(dim=1),
            (epe < 3).float().mean(dim=1),
            (epe < 5).float().mean(dim=1),
            torch.tensor(rel_error).cuda().repeat(epe.shape[0]),
            torch.tensor(abs_error).cuda().repeat(epe.shape[0]),
        ],
        dim=1
    )
    
    # (B // N_GPU, N_Metrics)
    return metrics


def merge_metrics(metrics):
    metrics = metrics.mean(dim=0)
    metrics = {
        'epe': metrics[0].item(),
        '1px': metrics[1].item(),
        '3px': metrics[2].item(),
        '5px': metrics[3].item(),
        'rel': metrics[4].item(),
        'abs': metrics[5].item(),
    }

    return metrics
