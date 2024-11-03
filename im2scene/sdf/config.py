from im2scene.encoder.psp_encoders import GradualStyleEncoder
from im2scene.sdf import models
from im2scene.sdf.models import sdf_model
from im2scene.sdf.models.sdf_utils import accumulate
from im2scene.giraffe.models import autoencoder


def get_model(cfg, device=None, len_dataset=0, args=None, **kwargs):
    # 使用sdf网络框架
    opt = args.sdf_opt
    # 定义模型
    if args.vae == 1:
        encoder = autoencoder.Encoder(img_size=opt.model.size, channel_in=3, z_size=opt.model.style_dim)
    elif args.psp == 1:
        encoder = GradualStyleEncoder(50, 'ir_se')
    else:
        encoder = None

    if args.need_train_vol_render:
        discriminator = sdf_model.VolumeRenderDiscriminator(opt.model).to(device)
        generator = sdf_model.Generator(opt.model, opt.rendering, full_pipeline=False).to(device)
        generator_test = sdf_model.Generator(opt.model, opt.rendering, ema=True, full_pipeline=False).to(device)
    else:
        discriminator = sdf_model.Discriminator(opt.model).to(device)
        generator = sdf_model.Generator(opt.model, opt.rendering).to(device)
        generator_test = sdf_model.Generator(opt.model, opt.rendering, ema=True).to(device)

    generator_test.eval()
    accumulate(generator_test, generator, 0)  # 生成器权重的指数移动平均

    model = models.SDFModel(
        device=device, encoder=encoder,
        discriminator=discriminator, generator=generator,
        generator_test=generator_test, )
    return model
