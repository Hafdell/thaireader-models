# thaireader-models

Model files used by ThaiReader, kept here so builds can download them at a
pinned URL and verify them by checksum. Each file is attached to a release,
not committed.

## `w2v-th.int8.onnx` (release `v1`)

A Thai speech model used for **forced alignment**: given a recording and its
known text, it finds when each letter is said, so the reader can highlight
words in time with narration.

| | |
|---|---|
| Base model | [`airesearch/wav2vec2-large-xlsr-53-th`](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th), revision `3155938c549b23eee16b1d4b55dcb161b7fe4bcf` |
| Changes | exported to ONNX (opset 17); weights quantized to 8-bit integers (`onnxruntime.quantization.quantize_dynamic`, QInt8) |
| Size | 317 MB |
| sha256 | `4b8cbcb695db1cbf7c9a972c35c847b028c3bf36bbc5fe83bad06302a67e3c4f` |
| Licence | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as the base model |

### Credit

The base model is by **AIResearch.in.th** (VISTEC-depa AI Research Institute of
Thailand): `facebook/wav2vec2-large-xlsr-53` fine-tuned on Thai Common Voice
7.0 — see their [model card](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th)
and [training code](https://github.com/vistec-ai/wav2vec2-large-xlsr-53-th).
This file is an adaptation of that work and is shared under the same licence.

### How it was made

With torch 2.15.0, transformers 5.22.2, onnx 1.23.2, onnxruntime 1.31.1 and
onnxscript 0.7.2 (these versions reproduce the file byte for byte):

```python
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "airesearch/wav2vec2-large-xlsr-53-th",
    revision="3155938c549b23eee16b1d4b55dcb161b7fe4bcf",
).eval()
torch.onnx.export(
    model, (torch.zeros(1, 16000),), "w2v-th.onnx",
    input_names=["input_values"], output_names=["logits"],
    dynamic_axes={"input_values": {1: "n"}, "logits": {1: "t"}},
    opset_version=17, dynamo=False,
)
quantize_dynamic("w2v-th.onnx", "w2v-th.int8.onnx", weight_type=QuantType.QInt8)
```

Input: mono 16 kHz audio, zero-mean / unit-variance normalized, shape
`[1, samples]`. Output: CTC logits `[1, frames, 70]` (one frame per 20 ms),
over the base model's `vocab.json`; `[PAD]` (id 69) is the blank.
