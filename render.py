import torch
import os
import argparse
from im2scene import config
from im2scene.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(
    description='Render images of a GIRAFFE model.'
)
parser.add_argument('--config', type=str, help='Path to config file.')
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
# 是否使用NeXT transformer
parser.add_argument("--next", type=int, default=0,
                    help='set 1 for use vae, 0 for default')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
if not os.path.exists(render_dir):
    os.makedirs(render_dir)

# Model
model = config.get_model(cfg, device=device, args=args)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
renderer = config.get_renderer(model, cfg, device=device)

model.eval()
if args.vae == 1:
    # 使用vae
    cfg['data']['path'] = r"data/celeba/render_test/*.jpg"  # 读取测试图像路径
    dataset = config.get_dataset(cfg)
    batch_size = 6
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=True,
        pin_memory=True, drop_last=True,
    )
    for batch in dataloader:
        x_real = batch.get('image').to(device)  # 读取真实图片
        out = renderer.render_full_visualization(
            render_dir,
            cfg['rendering']['render_program'],
            batch_size=batch_size,
            data=x_real)
else:
    out = renderer.render_full_visualization(
        render_dir,
        cfg['rendering']['render_program'])
