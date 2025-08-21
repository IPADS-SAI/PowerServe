#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from gguf.gguf_reader import GGUFReader
import numpy as np
VERSION = 20250821


class RopeType:
    NONE = -1
    NORMAL = 0
    NEOX = 2


def get_rope_type(arch: str) -> int:
    if arch == "llama":
        return RopeType.NORMAL
    elif arch in ("qwen", "phi3"):
        return RopeType.NEOX
    return RopeType.NONE


def field_to_value(field):
    """把 field 转成 Python 原生类型，自动处理字符串"""
    value = field.parts[field.data[0]]

    # 如果是 numpy 数组（或 memmap），尝试转换
    if isinstance(value, (np.ndarray, np.memmap)):
        # 如果是 uint8，可以解码成字符串
        if value.dtype == np.uint8:
            try:
                return value.tobytes().decode("utf-8")
            except Exception:
                return value.tolist()
        else:
            value = value.tolist()
            assert len(value)==1
            return value[0]
    else:
        raise TypeError()



def collect_config(reader: GGUFReader, config: dict):
    def get_val(key, required=True, default=None):
        if key not in reader.fields:
            if required:
                raise KeyError(f"Missing key: {key}")
            return default
        return field_to_value(reader.fields[key])

    def get_u32(key, required=True, default=0):
        val = get_val(key, required, default)
        return int(val) if val is not None else default

    def get_f32(key, required=True, default=0.0):
        val = get_val(key, required, default)
        return float(val) if val is not None else default

    def get_str(key, required=True, default=""):
        val = get_val(key, required, default)
        return str(val) if val is not None else default

    def get_arch_key(fmt: str) -> str:
        return fmt.format(model_arch)

    # architecture
    model_arch = get_str("general.architecture")
    config["model_arch"] = model_arch
    config["model_name"] = get_str("general.name", False, "unknown")

    # dims
    config["embed_dim"] = get_u32(get_arch_key("{}.embedding_length"))
    config["ffn_dim"] = get_u32(get_arch_key("{}.feed_forward_length"))
    config["n_attn_heads"] = get_u32(get_arch_key("{}.attention.head_count"))
    config["n_attn_kv_heads"] = get_u32(get_arch_key("{}.attention.head_count_kv"))
    config["n_layers"] = get_u32(get_arch_key("{}.block_count"))
    config["n_ctx"] = get_u32(get_arch_key("{}.context_length"))
    config["head_size"] = get_u32(get_arch_key("{}.attention.key_length"), default=config["embed_dim"] // config["n_attn_heads"])
    config["kv_dim"] = config["head_size"] * config["n_attn_kv_heads"]

    # vocab_size
    if get_arch_key("{}.vocab_size") in reader.fields:
        config["vocab_size"] = get_u32(get_arch_key("{}.vocab_size"))
    elif "tokenizer.ggml.tokens" in reader.fields:
        config["vocab_size"] = len(reader.fields["tokenizer.ggml.tokens"].data)
    else:
        raise KeyError("Cannot determine vocab size")

    # norm_eps
    if get_arch_key("{}.attention.layer_norm_epsilon") in reader.fields:
        config["norm_eps"] = get_f32(get_arch_key("{}.attention.layer_norm_epsilon"))
    elif get_arch_key("{}.attention.layer_norm_rms_epsilon") in reader.fields:
        config["norm_eps"] = get_f32(get_arch_key("{}.attention.layer_norm_rms_epsilon"))
    else:
        config["norm_eps"] = 1e-5

    # rope
    config["rope_dim"] = get_u32(get_arch_key("{}.rope.dimension_count"), False, config["head_size"])
    config["rope_freq_base"] = get_f32(get_arch_key("{}.rope.freq_base"), False, 10000.0)
    config["rope_scale_type"] = get_str(get_arch_key("{}.rope.scaling.type"), False, "linear")
    config["rope_attn_factor"] = get_f32(get_arch_key("{}.rope.scaling.attn_factor"), False, 1.0)
    config["n_rope_ctx_orig"] = get_u32(get_arch_key("{}.rope.scaling.original_context_length"), False, config["n_ctx"])

    rope_scale = 0.0
    if get_arch_key("{}.rope.scaling.factor") in reader.fields:
        rope_scale = get_f32(get_arch_key("{}.rope.scaling.factor"))
    elif get_arch_key("{}.rope.scale_linear") in reader.fields:
        rope_scale = get_f32(get_arch_key("{}.rope.scale_linear"))

    rope_scale = 1.0 if rope_scale == 0.0 else 1.0 / rope_scale
    config["rope_freq_scale"] = rope_scale
    config["rope_type"] = get_rope_type(model_arch)


def main():
    parser = argparse.ArgumentParser(description="Config Generator")
    parser.add_argument("--file-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--target-path", required=True, help="Path to output JSON config")
    args = parser.parse_args()

    print("This tool is buggy! You still need to modify the model.json by yourself.")

    reader = GGUFReader(Path(args.file_path))

    config = {"version": VERSION}
    collect_config(reader, config)

    with open(args.target_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
