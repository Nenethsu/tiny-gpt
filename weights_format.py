"""The file format the browser loads: model.json (config, vocabulary, tensor table) + model.bin (raw bytes).

Pure Python on purpose (no torch, no numpy), so the same code writes the real model and the
self-test model, and the browser engine can be checked against it on any machine.

Matrices are stored as int8, one float32 scale per row (symmetric per-row quantization):
    weight[r][c] ~= int8[r][c] * scale[r]
That makes the download about 4x smaller than float32. 1-D tensors (biases, LayerNorm) stay float32.
All numbers are little-endian. Every block starts on a 16-byte boundary so the browser can view
it directly as a Float32Array / Int8Array.
"""

import json
import os
import struct
import sys
from array import array

FORMAT = "tinygpt-v1"
ALIGN = 16


def quantize_rows(flat, rows, cols):
    """Returns (int8 values as bytes, float32 scales as bytes)."""
    q = array("b")
    scales = array("f")
    for r in range(rows):
        row = flat[r * cols : (r + 1) * cols]
        peak = max(abs(v) for v in row)
        scale = peak / 127.0 if peak > 0 else 1.0
        scale = struct.unpack("<f", struct.pack("<f", scale))[0]  # round to float32 like the reader sees it
        scales.append(scale)
        for v in row:
            q.append(max(-127, min(127, round(v / scale))))
    return q.tobytes(), _f32_bytes(scales)


def _f32_bytes(values):
    a = array("f", values)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


class Writer:
    def __init__(self):
        self.blob = bytearray()
        self.tensors = {}

    def _put(self, data):
        while len(self.blob) % ALIGN:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob += data
        return offset

    def add(self, name, shape, flat, quantize):
        """flat: the tensor's values in row-major order (a list of floats)."""
        n = 1
        for d in shape:
            n *= d
        if len(flat) != n:
            raise ValueError(f"{name}: {len(flat)} values for shape {shape}")
        if quantize:
            if len(shape) != 2:
                raise ValueError(f"{name}: only 2-D tensors are quantized")
            qbytes, sbytes = quantize_rows(flat, shape[0], shape[1])
            self.tensors[name] = {
                "shape": list(shape),
                "dtype": "int8",
                "offset": self._put(qbytes),
                "scales": self._put(sbytes),
            }
        else:
            self.tensors[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "offset": self._put(_f32_bytes(flat)),
            }

    def save(self, out_dir, config, vocab, eos_id, extra=None):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "model.bin"), "wb") as f:
            f.write(self.blob)
        manifest = {
            "format": FORMAT,
            "config": config,
            "vocab": vocab,
            "eos": eos_id,
            "bytes": len(self.blob),
            "tensors": self.tensors,
        }
        if extra:
            manifest.update(extra)
        with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True)
        return manifest


def read(out_dir):
    """Loads model.json + model.bin and returns (manifest, {name: (shape, dequantized flat list)})."""
    with open(os.path.join(out_dir, "model.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(out_dir, "model.bin"), "rb") as f:
        blob = f.read()
    tensors = {}
    for name, t in manifest["tensors"].items():
        n = 1
        for d in t["shape"]:
            n *= d
        if t["dtype"] == "float32":
            flat = list(struct.unpack_from(f"<{n}f", blob, t["offset"]))
        else:
            rows, cols = t["shape"]
            q = struct.unpack_from(f"<{n}b", blob, t["offset"])
            scales = struct.unpack_from(f"<{rows}f", blob, t["scales"])
            flat = [q[i] * scales[i // cols] for i in range(n)]
        tensors[name] = (t["shape"], flat)
    return manifest, tensors


def tensor_names(n_layer):
    """Every tensor the browser needs, in file order. lm_head is tied to wte, so it isn't stored."""
    names = [("wte.weight", True), ("wpe.weight", True)]
    for i in range(n_layer):
        p = f"h.{i}."
        names += [
            (p + "ln_1.weight", False),
            (p + "ln_1.bias", False),
            (p + "attn.c_attn.weight", True),
            (p + "attn.c_attn.bias", False),
            (p + "attn.c_proj.weight", True),
            (p + "attn.c_proj.bias", False),
            (p + "ln_2.weight", False),
            (p + "ln_2.bias", False),
            (p + "mlp.c_fc.weight", True),
            (p + "mlp.c_fc.bias", False),
            (p + "mlp.c_proj.weight", True),
            (p + "mlp.c_proj.bias", False),
        ]
    names += [("ln_f.weight", False), ("ln_f.bias", False)]
    return names
