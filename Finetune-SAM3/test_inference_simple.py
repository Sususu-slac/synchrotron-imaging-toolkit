import os
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ========= 修改这里 =========
image_path = "/home/subo/sam3/pill_segmentation_sam3_dataset/val/images/test_image.jpg"

base_checkpoint_path = "/home/subo/sam3/sam3.pt"

trained_checkpoint_path = "/home/subo/sam3/experiments/pill_sam3_test/checkpoints/checkpoint.pt"

bpe_path = "/home/subo/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

prompt = "boy"
# ============================


def clean_key(k):
    """
    Remove common training/DDP prefixes.
    Example:
      module.backbone.xxx -> backbone.xxx
      model.backbone.xxx  -> backbone.xxx
    """
    prefixes = ["module.", "model.", "_orig_mod."]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def collect_tensor_dicts(obj, path="root"):
    """
    Recursively find possible state_dict dictionaries inside a checkpoint.
    """
    candidates = []

    if isinstance(obj, dict):
        tensor_count = sum(torch.is_tensor(v) for v in obj.values())
        if tensor_count > 100:
            candidates.append((path, obj))

        for k, v in obj.items():
            if isinstance(v, dict):
                candidates.extend(collect_tensor_dicts(v, path + "." + str(k)))

    return candidates


def load_finetuned_weights(model, checkpoint_path):
    print("Loading trained checkpoint:")
    print(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        print("Top-level checkpoint keys:")
        for k in ckpt.keys():
            print("  -", k)

    target_keys = set(model.state_dict().keys())
    candidates = collect_tensor_dicts(ckpt)

    print(f"Found {len(candidates)} possible tensor dict candidates.")

    best_name = None
    best_state = None
    best_score = -1

    for name, sd in candidates:
        cleaned = {}

        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue

            nk = clean_key(k)

            if nk in target_keys:
                cleaned[nk] = v

        score = len(cleaned)
        print(f"Candidate: {name}, matched tensors: {score}")

        if score > best_score:
            best_score = score
            best_name = name
            best_state = cleaned

    if best_state is None or best_score == 0:
        raise RuntimeError(
            "Could not find matching model weights in checkpoint. "
            "Please print checkpoint keys and check format."
        )

    print(f"Best candidate: {best_name}")
    print(f"Matched tensors: {best_score}")

    result = model.load_state_dict(best_state, strict=False)

    print("Manual weight loading finished.")
    print("Missing keys:", len(result.missing_keys))
    print("Unexpected keys:", len(result.unexpected_keys))

    if len(result.missing_keys) > 0:
        print("First 20 missing keys:")
        for k in result.missing_keys[:20]:
            print("  ", k)

    if len(result.unexpected_keys) > 0:
        print("First 20 unexpected keys:")
        for k in result.unexpected_keys[:20]:
            print("  ", k)

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(base_checkpoint_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")

    if not os.path.exists(trained_checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_checkpoint_path}")

    image = Image.open(image_path).convert("RGB")
    print("Image:", image_path)
    print("Image size:", image.size)
    print("Prompt:", prompt)

    print("Building base SAM3 model first...")
    model = build_sam3_image_model(
        checkpoint_path=base_checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
    )

    print("Loading fine-tuned weights manually...")
    model = load_finetuned_weights(model, trained_checkpoint_path)

    model.to(device)
    model.eval()



    processor = Sam3Processor(model)

    print("Running inference...")
    if device == "cuda":
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt,
            )
    else:
        with torch.no_grad():
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt,
            )
    # with torch.no_grad():
    #     inference_state = processor.set_image(image)
    #     output = processor.set_text_prompt(
    #         state=inference_state,
    #         prompt=prompt,
    #     )

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print("Inference finished.")
    print("Number of masks:", len(masks))
    print("Masks shape:", masks.shape if hasattr(masks, "shape") else type(masks))
    print("Boxes:", boxes)
    print("Scores:", scores)


if __name__ == "__main__":
    main()
