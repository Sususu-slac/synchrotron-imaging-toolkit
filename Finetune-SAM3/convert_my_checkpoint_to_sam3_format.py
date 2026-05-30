import torch
from pathlib import Path
from collections import OrderedDict
from contextlib import redirect_stdout


base_sam3_path = "/home/subo/sam3/sam3.pt"

my_checkpoint_path = "/home/subo/sam3/experiments/pill_sam3_test/checkpoints/checkpoint.pt"

output_checkpoint_path = "/home/subo/sam3/experiments/pill_sam3_test/checkpoints/checkpoint_sam3_format.pt"

output_report_path = "/home/subo/sam3/experiments/pill_sam3_test/checkpoints/convert_to_sam3_format_report.txt"


def main():
    print("=" * 80)
    print("Base SAM3 checkpoint:")
    print(base_sam3_path)

    print("=" * 80)
    print("My fine-tuned checkpoint:")
    print(my_checkpoint_path)

    print("=" * 80)
    print("Output converted checkpoint:")
    print(output_checkpoint_path)

    print("=" * 80)
    print("Loading base SAM3 checkpoint...")
    base_ckpt = torch.load(base_sam3_path, map_location="cpu")

    if not isinstance(base_ckpt, dict):
        raise RuntimeError("base sam3.pt is not a dict.")

    print("Base checkpoint keys:", len(base_ckpt))
    print("Base tensor count:", sum(torch.is_tensor(v) for v in base_ckpt.values()))

    print("=" * 80)
    print("Loading my checkpoint...")
    my_ckpt = torch.load(my_checkpoint_path, map_location="cpu")

    if not isinstance(my_ckpt, dict):
        raise RuntimeError("my checkpoint.pt is not a dict.")

    if "model" not in my_ckpt:
        raise RuntimeError("my checkpoint.pt does not contain key: 'model'.")

    my_model = my_ckpt["model"]

    if not isinstance(my_model, dict):
        raise RuntimeError("my checkpoint['model'] is not a dict.")

    print("My checkpoint top-level keys:")
    for k in my_ckpt.keys():
        print("  -", k)

    print("My model keys:", len(my_model))
    print("My model tensor count:", sum(torch.is_tensor(v) for v in my_model.values()))

    print("=" * 80)
    print("Converting...")

    # Start from base checkpoint so that tracker and any missing detector keys are preserved
    converted = OrderedDict()
    for k, v in base_ckpt.items():
        converted[k] = v

    matched = []
    shape_mismatch = []
    not_found = []
    non_tensor = []

    for k, v in my_model.items():
        if not torch.is_tensor(v):
            non_tensor.append(k)
            continue

        # Your fine-tuned keys look like:
        # backbone.xxx
        # Original SAM3 keys look like:
        # detector.backbone.xxx
        if k.startswith("detector."):
            target_key = k
        else:
            target_key = "detector." + k

        if target_key not in converted:
            not_found.append((k, target_key))
            continue

        if not torch.is_tensor(converted[target_key]):
            not_found.append((k, target_key))
            continue

        if tuple(converted[target_key].shape) != tuple(v.shape):
            shape_mismatch.append(
                (
                    k,
                    target_key,
                    tuple(v.shape),
                    tuple(converted[target_key].shape),
                )
            )
            continue

        converted[target_key] = v
        matched.append((k, target_key))

    print("Matched and replaced weights:", len(matched))
    print("Not found in base SAM3:", len(not_found))
    print("Shape mismatch:", len(shape_mismatch))
    print("Non-tensor entries:", len(non_tensor))

    print("=" * 80)
    print("Base key prefix summary after conversion:")
    prefix_count = {}
    for k, v in converted.items():
        if torch.is_tensor(v):
            prefix = k.split(".")[0]
            prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

    for prefix, count in sorted(prefix_count.items()):
        print(f"{prefix}: {count}")

    print("=" * 80)
    print("First 30 matched keys:")
    for i, (src, dst) in enumerate(matched[:30]):
        print(f"[{i}] {src}  -->  {dst}")

    if not_found:
        print("=" * 80)
        print("First 50 keys not found:")
        for i, (src, dst) in enumerate(not_found[:50]):
            print(f"[{i}] {src}  -->  {dst}")

    if shape_mismatch:
        print("=" * 80)
        print("First 50 shape mismatches:")
        for i, item in enumerate(shape_mismatch[:50]):
            src, dst, src_shape, dst_shape = item
            print(f"[{i}] {src}  -->  {dst}")
            print(f"    my shape:   {src_shape}")
            print(f"    base shape: {dst_shape}")

    print("=" * 80)
    print("Saving converted checkpoint...")
    Path(output_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_checkpoint_path)

    print("Saved converted checkpoint to:")
    print(output_checkpoint_path)

    print("=" * 80)
    print("Done.")


Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)

with open(output_report_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        main()

print("Converted checkpoint saved to:")
print(output_checkpoint_path)

print("Conversion report saved to:")
print(output_report_path)
