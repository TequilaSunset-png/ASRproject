import csv
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import HubertForCTC, Wav2Vec2Processor

PROJECT = Path("/scratch/work/ouy1/speech_recognition")
DATA_ROOT = PROJECT / "data/project"

EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-5

torch.manual_seed(42)

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), flush=True)

processor = Wav2Vec2Processor.from_pretrained(
    PROJECT / "artifacts/processor",
    local_files_only=True,
)

model = HubertForCTC.from_pretrained(
    PROJECT / "models/mhubert-147",
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    bos_token_id=None,
    eos_token_id=None,
    local_files_only=True,
)

model.freeze_feature_encoder()
model.config.apply_spec_augment = False

for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0

model.config.layerdrop = 0.0



model.to(device)


class GeoDataset(Dataset):
    def __init__(self, split):
        path = DATA_ROOT / "geo" / f"{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, sr = sf.read(
            DATA_ROOT / row["file"],
            dtype="float32",
        )

        return audio, row["transcript"]


def collate_batch(examples):
    audios, texts = zip(*examples)

    inputs = processor(
        list(audios),
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        return_attention_mask=False,
    )

    audio_lengths = torch.tensor(
        [len(audio) for audio in audios],
        dtype=torch.long,
    )

    return inputs.input_values, audio_lengths, list(texts)


small_train = torch.utils.data.Subset(
    GeoDataset("train"),
    list(range(6000)),
)

train_loader = DataLoader(
    small_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=0,
)

dev_loader = DataLoader(
    GeoDataset("dev"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0,
)

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98),
    eps=1e-8,
    weight_decay=0.0,
)

warmup_steps = 160

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(
            (step + 1) / warmup_steps, 1.0
        ),
    )

ctc_loss = torch.nn.CTCLoss(
    blank=processor.tokenizer.pad_token_id,
    reduction="mean",
    zero_infinity=False,
)

print(
    f"Train: {len(train_loader.dataset)}, "
    f"Dev: {len(dev_loader.dataset)}",
    flush=True,
)

def evaluate_dev(epoch):
    model.eval()
    references = []
    predictions = []

    with torch.no_grad():
        for inputs, audio_lengths, texts in dev_loader:
            logits = model(input_values=inputs.to(device)).logits
            predicted_ids = logits.argmax(dim=-1).cpu()

            output_lengths = model._get_feat_extract_output_lengths(
                audio_lengths
            )

            for ids, length in zip(predicted_ids, output_lengths):
                prediction = processor.tokenizer.decode(
                    ids[:int(length)].tolist(),
                    group_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                predictions.append(prediction)

            references.extend(texts)

    print(f"\n===== Epoch {epoch} dev results =====", flush=True)
    print(f"WER: {100 * jiwer.wer(references, predictions):.2f}%", flush=True)
    print(f"CER: {100 * jiwer.cer(references, predictions):.2f}%", flush=True)

    for reference, prediction in zip(references[:5], predictions[:5]):
        print("\nREF:", reference, flush=True)
        print("HYP:", repr(prediction), flush=True)
    model.train()
    return references, predictions


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for step, (inputs, audio_lengths, texts) in enumerate(
        train_loader, start=1
    ):
        inputs = inputs.to(device)

        encoded = processor.tokenizer(
            texts,
            add_special_tokens=False,
        ).input_ids

        target_lengths = torch.tensor(
            [len(ids) for ids in encoded],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [token for ids in encoded for token in ids],
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_values=inputs).logits

        output_lengths = model._get_feat_extract_output_lengths(
            audio_lengths
        ).to(dtype=torch.long)

        loss = ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            targets,
            output_lengths,
            target_lengths,
        )


        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss: epoch={epoch}, step={step}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        
        '''
        if step == 1 or step % 100 == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Mean loss: {total_loss / step:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        '''

    if epoch % 10 == 0:
        model.eval()
        debug_refs = []
        debug_hyps = []
        blank_frames = 0
        total_frames = 0

        with torch.no_grad():
            for inputs, audio_lengths, texts in train_loader:
                logits = model(
                    input_values=inputs.to(device)
                ).logits

                ids = logits.argmax(dim=-1).cpu()

            
                blank_frames += (
                    ids == processor.tokenizer.pad_token_id
                ).sum().item()
                total_frames += ids.numel()

                debug_hyps.extend(
                    processor.batch_decode(ids)
                )
                debug_refs.extend(texts)

        print(
            f"[DEBUG epoch {epoch}] "
            f"train CER={jiwer.cer(debug_refs, debug_hyps):.3f}, "
            f"blank ratio={blank_frames / total_frames:.3f}",
            flush=True,
        )
        evaluate_dev(epoch)
        for ref, hyp in zip(debug_refs[:2], debug_hyps[:2]):
            print("REF:", ref, flush=True)
            print("HYP:", repr(hyp), flush=True)

        model.train()


model_dir = PROJECT / "outputs/baseline_final"
model_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(model_dir, safe_serialization=True)
processor.save_pretrained(model_dir)

print("Final model saved to:", model_dir, flush=True)
