import argparse
import os
from pathlib import Path

from common import REPO_ROOT, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Base model on a prepared replay dataset.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default="Salesforce/codet5p-770m")
    parser.add_argument("--tokenizer-name", default="Salesforce/codet5p-770m")
    parser.add_argument("--encoder-block-size", type=int, default=512)
    parser.add_argument("--decoder-block-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--python-bin", default="python")
    args = parser.parse_args()

    script = REPO_ROOT / "study-3" / "scripts" / "train_codet5+_Baseline.py"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.makedirs(args.base_output_dir, exist_ok=True)

    cmd = [
        args.python_bin,
        str(script),
        "--train-file",
        str(args.train_file),
        "--valid-file",
        str(args.valid_file),
        "--test-file",
        str(args.test_file),
        "--base-output-dir",
        str(args.base_output_dir),
        "--model-name-or-path",
        args.model_name_or_path,
        "--tokenizer-name",
        args.tokenizer_name,
        "--encoder-block-size",
        str(args.encoder_block_size),
        "--decoder-block-size",
        str(args.decoder_block_size),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--adam-epsilon",
        str(args.adam_epsilon),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--num-beams",
        str(args.num_beams),
    ]
    run_cmd(cmd, cwd=REPO_ROOT / "study-3" / "scripts")


if __name__ == "__main__":
    main()
