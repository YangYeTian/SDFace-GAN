import os
import torch
from tqdm.auto import tqdm
import numpy as np
import plyfile
import skimage.measure


def compute_alpha(net, p, z_s, length=1):
    # [1,points,3]  sigma:[1,points]
    # net 输入： p:[batch, points, 3] , z:[batch, 1, 256]
    sigma = net.get_sigma(p, z_s)

    alpha = 1 - torch.exp(-sigma * length).view(p.shape[:-1])

    return alpha


@torch.no_grad()
def getDenseAlpha(net, grid_Size, aabb, z_s, step_Size, device="cpu"):
    """
    gridSize是一个3维元组，代表了体素网格在每个维度上的大小。
    aabb是一个2x3的张量，代表了整个模型的坐标范围。
    device是计算设备，例如cpu或gpu。
    stepSize是体素光线穿过体素时的步长。
    compute_alpha是计算某个点alpha值的函数。

    首先，代码使用torch.meshgrid函数生成了一个大小为gridSize的3维网格点，
    每个点对应了密集体素网格中的一个体素。然后，根据网格点计算出了对应的三维坐标点dense_xyz。
    接下来，对于每个格网中的点，循环调用compute_alpha函数，计算其对应的alpha值，最终将alpha值填充到一个3维张量中返回。
    同时，也将对应的坐标点dense_xyz也返回了。
    """

    # 生成采样点 [242, 430, 257, 3]
    samples = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, grid_Size[0]),
        torch.linspace(0, 1, grid_Size[1]),
        torch.linspace(0, 1, grid_Size[2]),
    ), -1).to(device)
    dense_xyz = aabb[0] * (1 - samples) + aabb[1] * samples

    alpha = torch.zeros_like(dense_xyz[..., 0])
    for i in range(grid_Size[0]):
        # compute_alpha 输入【points，3】的xyz点，输出【points】的alpha，再进行变形

        p = dense_xyz[i].view(-1, 3)  # [110510, 3]
        p = p.unsqueeze(0)  # [1, 110510, 3]

        alpha[i] = compute_alpha(net, p, z_s, length=step_Size).view((grid_Size[1], grid_Size[2]))
        # print(alpha[i].shape)  ([430, 257])

    # print(alpha.shape)  # ([242, 430, 257])
    return alpha


def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        ply_filename_out,
        bbox,
        level=0.5,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def export_mesh(net, grid_Size, ply_filename_out, z_s, device):
    """
    marching_cubes导出模型
    :param net: 计算sigma的网络
    :param gridSize: 输出网格的大小，是一个三维矩阵
    :param ply_filename_out: 输出网格存储路径
    :param z_s: 潜在编码
    :return:
    """

    bounding_box = torch.tensor([[-1.5373, -1.3903, -1.0001], [1.5373, 1.3903, 1.0001]]).to(device)
    step_size = torch.tensor([0.0027]).to(device)

    # z_s : [batch_size, 1, 256]，单词输入仅输入一个z_s [1, 1, 256]
    for i in range(z_s.shape[0]):
        # 计算单个对象的alpha
        z_s_i = z_s[i]  # 每个对象对应一个zs
        # print(z_s_i.shape)  ([1, 256])
        alpha = getDenseAlpha(net, grid_Size, bounding_box, z_s_i, step_size, device)
        # 生成mesh  需要先将 tensor 转换到 CPU ，因为 Numpy 是 CPU-only
        convert_sdf_samples_to_ply(alpha.cpu(), ply_filename_out, bbox=bounding_box.cpu(), level=0.005)
        break


if __name__ == '__main__':
    import torch
    import os
    import argparse
    from im2scene import config
    from im2scene.checkpoints import CheckpointIO

    """
    设置参数
    """

    parser = argparse.ArgumentParser(
        description='Render images of a GIRAFFE model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    # 设置位置x和d编码方式
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    # hash编码参数
    parser.add_argument("--finest_res", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size", type=int, default=19,
                        help='log2 of hashmap size')
    # 是否使用小型网络
    parser.add_argument("--small_net", type=int, default=0,
                        help='set 1 for small net, 0 for default')
    # 是否使用vae
    parser.add_argument("--vae", type=int, default=0,
                        help='set 1 for use vae, 0 for default')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    """
    读取模型
    """
    model = config.get_model(cfg, device=device, args=args)

    checkpoint_io = CheckpointIO(out_dir, model=model)
    # checkpoint_io.load(cfg['test']['model_file'])
    checkpoint_io.load('model.pt')

    gen = model.generator_test
    if gen is None:
        gen = model.generator
    gen.eval()

    decoder = gen.decoder

    grid_Size = torch.tensor([121, 215, 128]).to(device)
    ply_filename_out = "test.ply"
    batch_size = 6

    latent_codes = gen.get_latent_codes(batch_size, tmp=0.65)

    if args.vae == 1:
        from im2scene.giraffe.training import reparameterize
        # 使用vae
        cfg['data']['path'] = r"data/celeba/render_test/*.jpg"  # 读取测试图像路径
        dataset = config.get_dataset(cfg, None)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=8, shuffle=True,
            pin_memory=True, drop_last=True)
        for batch in dataloader:
            x_real = batch.get('image').to(device)  # 读取真实图片
            encoder = model.encoder
            mu, logvar = encoder(x_real)
            z = reparameterize(mu, logvar)

            # 获取编码器编码结果，维度为512
            z = z.unsqueeze(dim=1)  # 增加一个维度
            z_s = z[..., :256]

            export_mesh(decoder, grid_Size, ply_filename_out, z_s, device)
