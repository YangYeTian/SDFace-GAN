import torch
from torchvision import transforms as T


def smoothness(model, bounding_box, styles, device, sample_points=32, voxel_size=0.1, margin=0.05, color=False):
    '''
    Smoothness loss of feature grid
    '''
    volume = bounding_box[:, 1] - bounding_box[:, 0]
    grid_size = (sample_points - 1) * voxel_size
    offset_max = bounding_box[:, 1] - bounding_box[:, 0] - grid_size - 2 * margin
    offset = torch.rand(3).to(offset_max) * offset_max + margin
    coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
    pts = (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size + bounding_box[:, 0] + offset

    pts_tcnn = (pts - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
    pts_tcnn = pts_tcnn.to(device)

    sdf = model.renderer.network.query_sdf(pts_tcnn, styles)

    tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
    tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
    tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()
    loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)
    return loss


def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    # x, y, z = torch.meshgrid(x, y, z, indexing="ij")
    x, y, z = torch.meshgrid(x, y, z)
    if not flatten:
        return torch.stack([x, y, z], dim=-1)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))



def normalize_3d_coordinate(p, bound):
    """
    Normalize 3d coordinate to [-1, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p


if __name__ == '__main__':
    # near = torch.tensor([[[-1]], [[-1]], [[-1]]])
    # far = torch.tensor([[[1]], [[1]], [[1]]])
    near = torch.tensor([[[-1.0]], [[-1.3]], [[-1.7]]])
    far = torch.tensor([[[7.0]], [[3.7]], [[1.4]]])
    # near = torch.tensor([[[0.8800]], [[0.8800]], [[0.8800]]])
    # far = torch.tensor([[[1.1200]], [[1.1200]], [[1.1200]]])
    bounding_box = torch.cat((near, far), dim=1).squeeze()

    smoothness(None, bounding_box, None, "cpu")
    # print(bounding_box)

