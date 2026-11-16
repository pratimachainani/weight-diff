# weight-diff

Compare a model checkpoint with its declared base, neuron by neuron, and see whether the changes are spread out (a normal fine-tune or quantization) or concentrated in a few rows (a reason to stop and audit).

Goes with the Agentic Ajrak video **"Claude Code found an AI backdoor hidden in 7 neurons"** (youtube.com/@the.agenticajrak), about Microsoft Research's Lazarus paper (10 Nov 2026): rewriting fewer than 10 dormant neurons can backdoor an agent model, and the model still passes its evals.

## Run it (Colab or any machine with RAM for two copies of the model)
```
pip install -q torch safetensors huggingface_hub
python weight_diff.py <base_model_id> <other_model_id>
python weight_diff.py <base_model_id> <other_model_id> --spike 7   # demo only
```
`--spike` pushes 7 random rows of the last layer's MLP in memory, to show what a concentrated change looks like. It is not a backdoor and writes nothing to disk. Embedding and output-head rows are skipped by default (new chat tokens make them noisy); add `--include-embeddings` to include them.

## Read the result
- DIFFUSE: changes are spread out. Not proof the model is safe.
- CONCENTRATED: a few rows moved far more than the rest. Ask who made the checkpoint, what they changed, and check the original publisher's hash. Some honest edits (merged adapters on a few layers, targeted knowledge edits) are concentrated too.

## Limits
A careful attacker will make the change as small as possible. This catches the pattern Microsoft describes, not every attack. Pair it with least privilege, a sandbox with outbound network limits, and human approval for high-impact agent actions.

Sources: Microsoft Security blog, "Detecting sleeper agents hiding in AI model neurons" (10 Nov 2026); Zeng, Severi, Bullwinkel, Horvitz, Papailiopoulos, "Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents" (Microsoft Research, 10 Nov 2026).

by Pratima Chainani · @the.agenticajrak · agenticajrak.com · MIT licence
