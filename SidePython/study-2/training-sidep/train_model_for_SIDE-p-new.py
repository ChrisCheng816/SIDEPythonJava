import argparse
import json
import logging
import os

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util
from torch.utils.data import DataLoader
from tqdm import tqdm


BEST_SCORE = float("-inf")
NO_IMPROVEMENT = 0


def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [InputExample(texts=[item["query"], item["pos"], item["neg"]]) for item in data]


def evaluate_val_score(model, val_data):
    scores = []
    for example in tqdm(val_data, desc="Evaluating"):
        query_emb = model.encode(example.texts[0], convert_to_tensor=True)
        pos_emb = model.encode(example.texts[1], convert_to_tensor=True)
        neg_emb = model.encode(example.texts[2], convert_to_tensor=True)

        cs_pos = util.cos_sim(query_emb, pos_emb).item()
        cs_neg = util.cos_sim(query_emb, neg_emb).item()
        scores.append(cs_pos - cs_neg)
    return sum(scores) / len(scores)


def main():
    global BEST_SCORE, NO_IMPROVEMENT

    parser = argparse.ArgumentParser(description="Train SIDE-py with triplet loss.")
    parser.add_argument("--train-file", default="side_finetune_codexglue.json")
    parser.add_argument("--val-file", default="side_finetune_codexglue_valid.json")
    parser.add_argument("--output-path", default="models/mpnet_triplet_no_hardneg_v2-test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint-steps", type=int, default=5000)
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.makedirs(args.output_path, exist_ok=True)

    log_file = os.path.join(args.output_path, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler(), file_handler],
    )

    train_examples = load_data(args.train_file)
    val_examples = load_data(args.val_file)
    print(len(train_examples), "train examples loaded")
    print(len(val_examples), "validation examples loaded")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    word_embedding_model = models.Transformer(
        "microsoft/mpnet-base", max_seq_length=args.max_seq_length
    )
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_loss = losses.TripletLoss(model=model)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    for epoch in range(args.epochs):
        logging.info("\n======== Epoch %s / %s ========", epoch + 1, args.epochs)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            output_path=None,
            checkpoint_path=args.output_path,
            checkpoint_save_steps=args.checkpoint_steps,
            show_progress_bar=True,
        )

        val_score = evaluate_val_score(model, val_examples)
        logging.info("Validation Score after epoch %s: %.4f", epoch + 1, val_score)

        if val_score > BEST_SCORE:
            BEST_SCORE = val_score
            NO_IMPROVEMENT = 0
            model.save(args.output_path)
            logging.info("New best model saved with score: %.4f", BEST_SCORE)
        else:
            NO_IMPROVEMENT += 1
            logging.info("No improvement. Patience counter: %s/%s", NO_IMPROVEMENT, args.patience)

        if NO_IMPROVEMENT >= args.patience:
            logging.info("Early stopping triggered.")
            break

    logging.info("\nBest validation score: %.4f", BEST_SCORE)


if __name__ == "__main__":
    main()
