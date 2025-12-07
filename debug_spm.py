import sentencepiece as spm
import os

model_file = "/home/kymy/Documents/python/mt-system/models/spm.model"
sp = spm.SentencePieceProcessor()
try:
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
    else:
        sp.load(model_file)
        print(f"BOS: {sp.bos_id()}")
        print(f"EOS: {sp.eos_id()}")
        print(f"UNK: {sp.unk_id()}")
        print(f"PAD: {sp.pad_id()}")

        print(f"ID 0: {sp.id_to_piece(0)}")
        print(f"ID 1: {sp.id_to_piece(1)}")
        print(f"ID 2: {sp.id_to_piece(2)}")
        print(f"ID 3: {sp.id_to_piece(3)}")
except Exception as e:
    print(f"Error: {e}")
