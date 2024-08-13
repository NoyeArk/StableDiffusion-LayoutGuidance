import os
import json
import torch
import hydra
from sdxl import StableDiffusionXLPipeline
from omegaconf import OmegaConf
from utils import load_text_inversion
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from utils import compute_ca_loss, Phrase2idx, draw_box, setup_logger


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):
    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)

    print('inference中main初始化')
    from diffusers.models.unets import unet_2d_condition
    # unet = unet_2d_condition_XL.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path,
    #                                                                                 subfolder="unet")
    # unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path,
    #                                                                                 subfolder="unet")
    # print('+++++++++++++++++++++++++')
    # tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    # if cfg.general.real_image_editing:
    #     text_encoder, tokenizer = load_text_inversion(text_encoder, tokenizer, cfg.real_image_editing.placeholder_token,
    #                                                   cfg.real_image_editing.text_inversion_path)
    #     unet.load_state_dict(torch.load(cfg.real_image_editing.dreambooth_path)['unet'])
    #     text_encoder.load_state_dict(torch.load(cfg.real_image_editing.dreambooth_path)['encoder'])
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unet.to(device)
    # text_encoder.to(device)
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

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.to("cuda:7")

    # 推理
    pil_images = pipe(
        prompt=examples['prompt'],
        vae=vae,
        bboxes=examples['bboxes'],
        phrases=examples['phrases'],
        cfg=cfg,
        logger=logger
    )

    # 保存示例图片
    for index, pil_image in enumerate(pil_images):
        image_path = os.path.join(cfg.general.save_path, 'example_{}.png'.format(index))
        logger.info('save example image to {}'.format(image_path))
        draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)


if __name__ == "__main__":
    main()
