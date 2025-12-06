import sentencepiece as spm
import argparse
import os

def main():
    """
    Trains a SentencePiece model using the Python library.
    This script is a wrapper around the spm.SentencePieceTrainer.train() method,
    allowing you to pass command-line arguments just like the spm_train executable.
    """
    # Khởi tạo parser để xử lý các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Train a SentencePiece model from a Python script.")

    # Định nghĩa các tham số tương tự như spm_train
    parser.add_argument('--input', type=str, required=True, help='Input text file.')
    parser.add_argument('--model_prefix', type=str, required=True ,help='Prefix for the model name.')
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size.')
    parser.add_argument('--character_coverage', type=float, default=1.0, help='Character coverage.')
    parser.add_argument('--model_type', type=str, default='unigram', help='Model type (e.g., unigram, bpe).')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Directory where the input file is located.')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the trained model.')

    # Xử lý các tham số được truyền vào
    args = parser.parse_args()

    # Tạo thư mục output nếu nó chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)

    # Xây dựng chuỗi lệnh cho SentencePiece trainer
    input_path = os.path.join(args.data_dir, args.input)
    model_path_prefix = os.path.join(args.output_dir, args.model_prefix)
    # Hàm spm.SentencePieceTrainer.train() nhận một chuỗi các tham số
    command = (
        f'--input={input_path} '
        f'--model_prefix={model_path_prefix} '
        f'--vocab_size={args.vocab_size} '
        f'--character_coverage={args.character_coverage} '
        f'--model_type={args.model_type}'
    )

    print("Starting SentencePiece training with command:")
    print(f"spm.SentencePieceTrainer.train('{command}')")

    # Gọi hàm training của SentencePiece
    spm.SentencePieceTrainer.train(command) # type: ignore

    print(f"Training complete. Model and vocab files are saved in '{args.output_dir}' with prefix '{args.model_prefix}'.")

if __name__ == '__main__':
    main()
