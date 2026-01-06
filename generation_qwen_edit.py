species_list = ["Coccinellidae septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris", "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris", "Eristalis tenax"]
species_dict = {
    "Coccinellidae septempunctata": "Common Ladybug",
    "Apis mellifera": "Western Honey Bee",
    "Bombus lapidarius": "Red-tailed bumblebee",
    "Bombus terrestris": "Buff-tailed bumblebee",
    "Eupeodes corollae": "Migrant Hoverfly",
    "Episyrphus balteatus": "Marmalade Hoverfly",
    "Aglais urticae": "Small Tortoiseshell Butterfly",
    "Vespula vulgaris": "Common wasp",
    "Eristalis tenax": "Common Drone Fly"
}

# load model
import math
import torch
import random
import os
from pathlib import Path
from PIL import Image
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler

# ----------------------------
# SETUP PIPELINE
# ----------------------------
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"
)

import torch
import random
from pathlib import Path
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
INPUT_ROOT = Path("/data/vision/beery/scratch/serena/insect_analysis/BJERGE_NEW/prepared_GBIF_BJERGE/train")
OUTPUT_ROOT = Path("/data/vision/beery/scratch/serena/diffusion/generated_full_dataset_qwen_edit")
NEW_IMAGES_PER_CLASS = 500   # generate this many additional images

VIEW_ANGLES = [
    "top-down",
    "side view",
    "three-quarter lateral view"
]

BASE_PROMPT_TEMPLATE = """
Edit the input image of {species_name} ({common_name}) to appear as a highly realistic photograph.
Preserve the identity, scale, and diagnostic traits of the insect, including correct wing venation,
body proportions, and coloration.

The insect should be shown in a {view_angle} pose on a sedum flowerbed.
Use natural, real-world lighting with realistic shadows.
Maintain anatomical accuracy and do not introduce artifacts.
Do not add or remove objects, and do not alter the species identity.
"""

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# EDITING LOOP
# ----------------------------
for species_folder in INPUT_ROOT.iterdir():
    if not species_folder.is_dir():
        continue

    species_name = species_folder.name
    if species_name == "Bombus terrestris" or species_name == 'Bombus lapidarius' or species_name == 'Coccinellidae septempunctata':
        continue

    common_name = species_dict.get(species_name, species_name)
    output_species_folder = OUTPUT_ROOT / species_name
    output_species_folder.mkdir(parents=True, exist_ok=True)

    input_images = list(species_folder.glob("*.[jp][pn]g"))
    if not input_images:
        continue

    # Count existing generated images
    existing_images = sorted(output_species_folder.glob("qwen-edit-*.jpg"))
    start_idx = len(existing_images)
    end_idx = start_idx + NEW_IMAGES_PER_CLASS

    for i in range(start_idx, end_idx):
        out_path = output_species_folder / f"qwen-edit-{species_name}-{i:04d}.jpg"

        init_image = Image.open(random.choice(input_images)).convert("RGB")
        view_angle = random.choice(VIEW_ANGLES)

        prompt = BASE_PROMPT_TEMPLATE.format(
            species_name=species_name,
            common_name=common_name,
            view_angle=view_angle
        )

        try:
            edited_image = pipe(
                image=init_image,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=10,
                true_cfg_scale=2.0,
                generator=torch.Generator(device=pipe.device).manual_seed(
                    random.randint(0, 1_000_000)
                )
            ).images[0]

            edited_image.save(out_path, format="JPEG", quality=95)

        except Exception:
            continue

    print(f"[DONE] Added {NEW_IMAGES_PER_CLASS} images for {species_name} "
          f"(total now: {end_idx})")

print("[ALL DONE] Additional image generation complete")
