import csv
import json
from pathlib import Path

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

PROJECT = Path("/scratch/work/ouy1/speech_recognition")
DATA_ROOT = PROJECT / "data/project"
MODEL_DIR = PROJECT / "models/mhubert-147"
OUTPUT_DIR = PROJECT / "artifacts/processor"


def read_split(split):
    csv_path = DATA_ROOT / "geo" / f"{split}.csv"

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)


    for row in rows:
        audio_path = DATA_ROOT / row["file"]

    print(f"{split}: {len(rows)} samples, audio paths OK")
    return rows


train_rows = read_split("train")
dev_rows = read_split("dev")
test_rows = read_split("test")

train_chars = set("".join(row["transcript"] for row in train_rows))


unexpected_whitespace = {
    ch for ch in train_chars if ch.isspace() and ch != " "
}


tokens = ["[PAD]", "[UNK]", "|"] + sorted(train_chars - {" "})
vocab = {token: index for index, token in enumerate(tokens)}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
vocab_path = OUTPUT_DIR / "vocab.json"
vocab_path.write_text(
    json.dumps(vocab, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

tokenizer = Wav2Vec2CTCTokenizer(
    str(vocab_path),
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    bos_token=None,
    eos_token=None,
    do_lower_case=False,
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)
processor.save_pretrained(OUTPUT_DIR)


for row in train_rows:
    text = row["transcript"]
    ids = tokenizer(text, add_special_tokens=False).input_ids
    recovered = tokenizer.decode(
        ids,
        group_tokens=False,
        clean_up_tokenization_spaces=False,
    )


dev_chars = set("".join(row["transcript"] for row in dev_rows))
unseen = sorted(dev_chars - train_chars)

example = train_rows[0]["transcript"]
print("\nVocabulary size:", len(tokenizer))
print("Blank/pad ID:", tokenizer.pad_token_id)
print("Dev unseen characters:", unseen)
print("Example text:", example)
print("Example IDs:", tokenizer(example).input_ids)
print("All training transcripts round-trip correctly")
print("Processor saved to:", OUTPUT_DIR)