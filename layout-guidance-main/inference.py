import os
import json
import torch
import hydra
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import load_text_inversion
from my_model import unet_2d_condition
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from utils import compute_ca_loss, Phrase2idx, draw_box, setup_logger


def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg, logger):
    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")

    # 获取对象位置
    logger.info("Convert Phrases to Object Positions")
    object_positions = Phrase2idx(prompt, phrases)

    # 编码分类器嵌入
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # 编码提示词
    input_ids = tokenizer(
        [prompt] * cfg.inference.batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    # 种子生成器，用于产生初始潜在噪声
    generator = torch.manual_seed(cfg.inference.rand_seed)

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start,
                                           beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule,
                                           num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0

        while loss.item() / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            # 使用unet进行预测，得到预测的噪声和注意力图
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings, added_cond_kwargs=added_cond_kwargs)

            # 使用指导更新潜伏物
            loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   object_positions=object_positions) * cfg.inference.loss_scal

            # 使用自动求导机制计算损失对 latents 的梯度
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            # 根据计算出的梯度和噪声调度器的参数更新 latents
            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2
            iteration += 1

            # 清理 CUDA 缓存以释放内存
            torch.cuda.empty_cache()

        # 禁用梯度计算
        with torch.no_grad():
            # 将 latents 张量复制一份并拼接在一起，形成一个新的张量 latent_model_input
            latent_model_input = torch.cat([latents] * 2)

            # 对输入进行缩放
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            # 获得噪声预测样本
            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (
                        noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):
    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)

    print('inference中main初始化')
    from diffusers.models.unets import unet_2d_condition
    # unet = unet_2d_condition_XL.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path,
    #                                                                                 subfolder="unet")
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path,
                                                                                    subfolder="unet")
    print('+++++++++++++++++++++++++')
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    if cfg.general.real_image_editing:
        text_encoder, tokenizer = load_text_inversion(text_encoder, tokenizer, cfg.real_image_editing.placeholder_token,
                                                      cfg.real_image_editing.text_inversion_path)
        unet.load_state_dict(torch.load(cfg.real_image_editing.dreambooth_path)['unet'])
        text_encoder.load_state_dict(torch.load(cfg.real_image_editing.dreambooth_path)['encoder'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

    # ------------------ 示例输入 ------------------
    examples = {"prompt": "A hello kitty toy is playing with a purple ball.",
                "phrases": "hello kitty; ball",
                "bboxes": [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],
                'save_path': cfg.general.save_path
                }

    # ------------------ 真实图像编辑示例输入 ------------------
    if cfg.general.real_image_editing:
        examples = {"prompt": "A {} is standing on grass.".format(cfg.real_image_editing.placeholder_token),
                    "phrases": "{}".format(cfg.real_image_editing.placeholder_token),
                    "bboxes": [[[0.4, 0.2, 0.9, 0.9]]],
                    'save_path': cfg.general.save_path
                    }
    # ---------------------------------------------------

    # 准备保存路径
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)

    logger.info(cfg)
    # Save cfg
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

    # 推理
    pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'],
                           examples['phrases'], cfg, logger)

    # 保存示例图片
    for index, pil_image in enumerate(pil_images):
        image_path = os.path.join(cfg.general.save_path, 'example_{}.png'.format(index))
        logger.info('save example image to {}'.format(image_path))
        draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)


if __name__ == "__main__":
    main()
