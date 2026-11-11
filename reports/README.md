# Experiment reports

Each folder here is one real training run, written by `python report.py` on the machine that ran it:

- `REPORT.md`: setup, models, evaluation table, sample stories, illustrations, systems measurements, limitations
- `manifest.json`: commit, library versions, hardware, data sizes, SHA-256 of every file, training settings
- `eval.json`: every story the evaluation generated, failures included
- `image_gpt.png`, `dit.png`: illustration samples, if those models were trained

Samples use fixed prompts and seeds; nothing is picked by hand. Checkpoints are not committed
(they are large); their checksums in `manifest.json` identify exactly which weights produced the report.
