# Eval Bundle Provenance

`heldout.txt` is a 100,574-byte held-out evaluation slice from the public TinyStories dataset:

- source dataset: `roneneldan/TinyStories`
- dataset card: <https://huggingface.co/datasets/roneneldan/TinyStories>
- dataset license listed by the card: `cdla-sharing-1.0`
- license text: <https://cdla.dev/sharing-1-0/>

Why this ships in the repo:

- the TinyStories dataset card marks the dataset as `cdla-sharing-1.0`
- CDLA-Sharing-1.0 section 3 allows publishing data or subsets under the same agreement as long as the agreement name and access method are preserved
- this repo preserves that attribution and license link here

Bundle contents:

- `heldout.txt` — TinyStories held-out slice used for bits-per-byte / NLL scoring
- `prompts.txt` — local prompt set used for sample generation
- `wordlist.txt` — local helper list used for the word-score metric

Checksums:

- `heldout.txt` sha256: `ecf5ded52ebcf662b749e475e21c5e2b5977df6c8a2fbec450209ff7fd565a46`
- `prompts.txt` sha256: `903fdcc3e3dc7745a5780864f26ea7d9f8130948a00eca3aec1eb9d20c1b4253`
- `wordlist.txt` sha256: `5b7063f0380115d205f57c48fabee2fae498501739356915b0e8c5bd72d51afd`
