# GSM8K data

These JSONL files are a deterministic conversion of the official GSM8K
release from [OpenAI's grade-school-math repository](https://github.com/openai/grade-school-math).
GSM8K is distributed under the MIT License; the upstream license is preserved
in `LICENSE-GSM8K`.

`gsm8k_validation.jsonl` contains 512 rows selected from the official training
split with Python's `random.Random(42)`. `gsm8k_train.jsonl` contains the other
6,961 training rows. `gsm8k_test.jsonl` preserves all 1,319 official test rows.
The worked solutions are intentionally omitted; each row contains only the
question prompt, final numeric label, and source metadata.

Regenerate the files with:

```bash
python3 examples/chimera/prepare_gsm8k.py
```

The converter pins and verifies the SHA256 digest of both official source
files before writing deterministic, compact JSONL output.

| File | Rows | SHA256 |
| --- | ---: | --- |
| `gsm8k_train.jsonl` | 6,961 | `cf0d40145360128d741197a5b8a0ee9c7a0c89ab4f80f8d044367293c740d464` |
| `gsm8k_validation.jsonl` | 512 | `6cd6c0582d4fecdd0d89a84b5342f9aa97d0c65290ce2d6c0dd9be1d4ad62959` |
| `gsm8k_test.jsonl` | 1,319 | `f18307e87d61caee9719518f587825cb603256cfb09af3ba326449e0bc81b9fc` |
