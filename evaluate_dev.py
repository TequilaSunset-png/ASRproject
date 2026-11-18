"""Evaluate a saved GEO CTC model without retraining it."""

import argparse
import csv
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from transformers import HubertForCTC, Wav2Vec2Processor


def main():
    project = Path('/scratch/work/ouy1/speech_recognition')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=project / 'outputs/baseline_final')
    parser.add_argument('--data-root', type=Path,
                        default=project / 'data/project')
    parser.add_argument('--output', type=Path,
                        default=project / 'dev_predictions_baseline.csv')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('Run this script in a Slurm GPU job.')
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)

    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Saved model:', args.model_dir, flush=True)
    # Load both components from the final saved directory, not the SSL model.
    processor = Wav2Vec2Processor.from_pretrained(
        args.model_dir, local_files_only=True,
    )
    model = HubertForCTC.from_pretrained(
        args.model_dir, local_files_only=True,
    ).cuda().eval()

    if model.lm_head.out_features != len(processor.tokenizer):
        raise ValueError('Model and tokenizer vocabulary sizes differ.')
    if model.config.pad_token_id != processor.tokenizer.pad_token_id:
        raise ValueError('Model and tokenizer blank IDs differ.')

    with (args.data_root / 'geo/dev.csv').open(
        encoding='utf-8-sig', newline='',
    ) as f:
        reader = csv.DictReader(f)
        if not {'file', 'transcript'}.issubset(reader.fieldnames or []):
            raise ValueError('dev.csv must contain file and transcript columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('dev.csv is empty.')

    references, predictions = [], []
    # One utterance at a time: no padding and no padded frames to trim.
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            audio_path = args.data_root / row['file']
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != 16000 or audio.ndim != 1 or len(audio) == 0:
                raise ValueError(f'Expected nonempty 16 kHz mono audio: {audio_path}')
            reference = row['transcript']
            if not reference or not reference.strip():
                raise ValueError(f'Missing reference: {row["file"]}')

            inputs = processor(
                audio, sampling_rate=sr, return_tensors='pt',
                return_attention_mask=False,
            ).input_values.cuda()
            logits = model(input_values=inputs).logits
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f'Non-finite logits: {row["file"]}')
            ids = logits.argmax(dim=-1)[0].cpu().tolist()
            prediction = processor.tokenizer.decode(
                ids, group_tokens=True, clean_up_tokenization_spaces=False,
            )
            # Score the original reference, preserving characters such as ﬁ.
            references.append(reference)
            predictions.append(prediction)
            if index == 1 or index % 100 == 0 or index == len(rows):
                print(f'Evaluated {index}/{len(rows)}', flush=True)

    # Write predictions before scoring so the inference results are retained.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reference', 'prediction'])
        writer.writerows(zip(references, predictions))
    print('\nPredictions saved to:', args.output, flush=True)

    print('\n===== Dev results =====', flush=True)
    print(f'WER: {100 * jiwer.wer(references, predictions):.2f}%', flush=True)
    print(f'CER: {100 * jiwer.cer(references, predictions):.2f}%', flush=True)
    empty = sum(not text.strip() for text in predictions)
    print(f'Empty predictions: {empty}/{len(predictions)}', flush=True)
    for reference, prediction in zip(references[:5], predictions[:5]):
        print('\nREF:', reference, flush=True)
        print('HYP:', repr(prediction), flush=True)


if __name__ == '__main__':
    main()
