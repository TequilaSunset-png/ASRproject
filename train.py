import csv
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import HubertForCTC, Wav2Vec2Processor

from training_checkpoint import parse_training_args, restore_checkpoint, save_checkpoint

args = parse_training_args(Path("/scratch/work/ouy1/speech_recognition"), "baseline")
PROJECT = args.project
DATA_ROOT = PROJECT / "data/project"

EPOCHS = args.epochs
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
        module.p = 0.1

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
        return_attention_mask=True,
    )

    return inputs.input_values, inputs.attention_mask, list(texts)


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
        for inputs, attention_mask, texts in dev_loader:
            logits = model(
                input_values=inputs.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            predicted_ids = logits.argmax(dim=-1).cpu()

            output_lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1)
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

    return references, predictions


# Checkpoints from the two freezing strategies are intentionally incompatible.
checkpoint_metadata = {
    "training_mode": "feature_encoder_frozen",
    "batch_size": BATCH_SIZE,
    "train_size": len(train_loader.dataset),
    "learning_rate": LEARNING_RATE,
    "warmup_steps": warmup_steps,
    "vocab": processor.tokenizer.get_vocab(),
}
start_epoch, global_step = 1, 0
if args.resume is not None:
    start_epoch, global_step = restore_checkpoint(
        args.resume, model, optimizer, scheduler, checkpoint_metadata,
    )

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for step, (inputs, attention_mask, texts) in enumerate(
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
        logits = model(
            input_values=inputs,
            attention_mask=attention_mask.to(device),
        ).logits

        output_lengths = model._get_feat_extract_output_lengths(
            attention_mask.sum(dim=-1)
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
        global_step += 1

        total_loss += loss.item()

    if epoch % 2 == 0:
        model.eval()
        debug_refs = []
        debug_hyps = []
        blank_frames = 0
        total_frames = 0

        with torch.no_grad():
            for inputs, attention_mask, texts in train_loader:
                logits = model(
                    input_values=inputs.to(device),
                    attention_mask=attention_mask.to(device),
                ).logits

                ids = logits.argmax(dim=-1).cpu()

                output_lengths = model._get_feat_extract_output_lengths(
                    attention_mask.sum(dim=-1)
                )
                for sequence, length in zip(ids, output_lengths):
                    valid_ids = sequence[:int(length)]
                    blank_frames += (
                        valid_ids == processor.tokenizer.pad_token_id
                    ).sum().item()
                    total_frames += valid_ids.numel()
                    debug_hyps.append(processor.tokenizer.decode(
                        valid_ids.tolist(),
                        group_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ))
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

    print(f"Epoch {epoch}: loss={total_loss / len(train_loader):.4f}", flush=True)
    if epoch % args.save_every == 0 or epoch == EPOCHS:
        save_checkpoint(
            args.checkpoint_dir / "latest.pt", model, optimizer, scheduler,
            epoch, global_step, checkpoint_metadata,
        )


model_dir = args.output_dir
model_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(model_dir, safe_serialization=True)
processor.save_pretrained(model_dir)

print("Final model saved to:", model_dir, flush=True)
