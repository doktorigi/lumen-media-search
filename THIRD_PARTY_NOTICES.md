# Third-Party Notices

Lumen is distributed under the MIT License in `LICENSE`. Its runtime and
development dependencies remain subject to their own licenses.

## CLIP ViT-B/32 OpenVINO model

Lumen downloads the following model at runtime; model files are not included in
the Lumen Python distributions:

- Repository: `scaleflex/clip-vit-base-patch32-openvino`
- Pinned revision: `26f5e9f1ec6185017f8a7422e63c8fc083403846`
- Publisher/converter: Scaleflex
- Original model: OpenAI CLIP ViT-B/32
- Repository license metadata: MIT

Source and model card:
https://huggingface.co/scaleflex/clip-vit-base-patch32-openvino

Lumen allowlists these runtime files and verifies them before inference:

| File | SHA-256 |
| --- | --- |
| `clip-vit-base-patch32.bin` | `a3ff68d119cd70f91edb3016f53804d6bc101b38229a45d1df2e66353c2d58a9` |
| `clip-vit-base-patch32.xml` | `0eaf9f5eb7fd76ec9788bcadc4f22a5bb06d40edd3bf31844618f6105ab77649` |
| `preprocessor_config.json` | `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd` |
| `special_tokens_map.json` | `f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf` |
| `tokenizer.json` | `b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842` |
| `tokenizer_config.json` | `cbcdcf14cede6b5c3a6841f37961364357d82bef3079beb115e95f27252526a0` |
| `vocab.json` | `5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363` |

Original implementation, license, and model card:

- https://github.com/openai/CLIP
- https://github.com/openai/CLIP/blob/main/LICENSE
- https://github.com/openai/CLIP/blob/main/model-card.md

The original CLIP license contains this notice:

> MIT License — Copyright (c) 2021 OpenAI

The model card describes CLIP as a research output rather than a model developed
for general deployment. It calls for careful, task-specific evaluation and
identifies limitations including weaker fine-grained classification and
counting, English-focused evaluation, and fairness and bias risks. Surveillance
and facial-recognition uses are outside the model's stated scope. Lumen does not
implement named-person or facial recognition; release evaluation should use
representative, in-domain media and queries.

## Mojo

Lumen includes an optional Mojo ranker source file. The Mojo SDK and Python
package are not bundled with Lumen and are not installed by Lumen's default
dependencies. Running `scripts/validate_mojo.sh` installs the separately
licensed `mojo==1.0.0b2` package into a user-local validation environment.

Mojo and MAX are governed by the Modular Community License and related terms:

- https://www.modular.com/legal/community
- https://www.modular.com/legal/terms
- https://www.modular.com/legal/trademark

MAX® and Mojo® are trademarks of Modular, Inc. used under license.
