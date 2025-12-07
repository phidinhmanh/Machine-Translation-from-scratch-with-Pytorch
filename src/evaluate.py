import torch
import argparse
import os
import matplotlib.pyplot as plt
import sacrebleu
import sentencepiece as spm
import google.generativeai as genai
from tqdm import tqdm
from datasets import load_dataset
from model import Transformer


# import getpass

# os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(GEMINI_API_KEY)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best_transformer.pth"
    )
    parser.add_argument("--spm_model", type=str, default="models/spm.model")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--beam_size", type=int, default=3, help="Beam size for decoding"
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument(
        "--test_samples",
        type=int,
        default=100,
        help="Số lượng câu test (đừng test hết nếu dùng Gemini)",
    )

    # Model params (Phải khớp với lúc train)
    parser.add_argument("--vocab_size", type=int, default=7000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff_expansion", type=int, default=4)
    parser.add_argument("--pad_idx", type=int, default=0)

    return parser.parse_args()


# --- 1. BEAM SEARCH DECODING (ENGINEER LEVEL) ---
def beam_search_decode(model, src, sp, device, beam_size=3, max_len=128):
    """
    Thuật toán Beam Search để tìm bản dịch tốt nhất.
    """
    model.eval()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # Encoder output chỉ cần tính 1 lần
    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        enc_out = model.dropout(model.pos_encoder(model.embedding(src)))
        for layer in model.transformer_encoder:
            enc_out = layer(enc_out, src_mask)
        # Final Norm encoder
        enc_out = model.final_norm(enc_out)

    # Khởi tạo Beam: Mỗi beam chứa (sequence, score)
    # Sequence bắt đầu bằng [BOS]
    k_candidates = [(torch.tensor([bos_id], dtype=torch.long, device=device), 0.0)]

    # Loop cho đến max_len
    for _ in range(max_len):
        new_candidates = []

        for seq, score in k_candidates:
            # Nếu câu đã kết thúc bằng EOS, giữ nguyên
            if seq[-1].item() == eos_id:
                new_candidates.append((seq, score))
                continue

            # Forward Decoder
            # Lưu ý: Pass seq shape [1, Len] vào
            tgt_input = seq.unsqueeze(0)
            tgt_mask = model.make_trg_mask(tgt_input)

            dec_out = model.dropout(model.pos_encoder(model.embedding(tgt_input)))
            for layer in model.transformer_decoder:
                dec_out = layer(dec_out, tgt_mask, context=enc_out, src_mask=src_mask)

            dec_out = model.final_norm(dec_out)
            out = model.linear(dec_out)  # [1, Len, Vocab]

            # Lấy xác suất của token cuối cùng
            # Dùng LogSoftmax để cộng điểm cho dễ (thay vì nhân xác suất)
            probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze()

            # Lấy top-k token tốt nhất tiếp theo
            topk_probs, topk_ids = torch.topk(probs, beam_size)

            for i in range(beam_size):
                token = topk_ids[i]
                prob = topk_probs[i].item()

                # Tạo sequence mới
                new_seq = torch.cat([seq, token.unsqueeze(0)], dim=0)
                new_score = score + prob  # Cộng log prob
                new_candidates.append((new_seq, new_score))

        # Sắp xếp tất cả candidates theo score giảm dần và lấy top k
        k_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[
            :beam_size
        ]

        # Nếu tất cả các beam đều đã gặp EOS thì dừng sớm
        if all(c[0][-1].item() == eos_id for c in k_candidates):
            break

    # Lấy sequence có điểm cao nhất
    best_seq = k_candidates[0][0]
    return best_seq.cpu().tolist()


# --- 2. GEMINI SCORE (LLM-AS-A-JUDGE) ---
def get_gemini_score(source, reference, candidate):
    """
    Dùng AI chấm điểm AI. Gửi prompt lên Google Gemini.
    Trả về điểm 0-100.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return None  # Bỏ qua nếu không có key

    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
    model = genai.GenerativeModel("models/gemini-2.5-flash")  # type: ignore

    prompt = f"""
    You are a professional translator. Evaluate the quality of the translation from English to Vietnamese.
    
    Source (English): "{source}"
    Reference (Vietnamese): "{reference}"
    Candidate (Machine Translation): "{candidate}"
    
    Score the Candidate translation on a scale from 0 to 100 based on accuracy, fluency, and meaning preservation.
    Return ONLY the number.
    """

    try:
        response = model.generate_content(prompt)
        score = int(response.text.strip())
        return score
    except Exception as e:
        print(f"Error: {e}")
        return 50  # Fallback nếu lỗi


# --- 3. MAIN EVALUATION LOOP ---
def main():
    args = get_args()

    # 1. Load Tokenizer & Data
    print("⏳ Loading Tokenizer & Data...")
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)  # type: ignore

    # Load test set (dùng tập test thật của opus100)
    dataset = load_dataset("opus100", "en-vi", split=f"test[:{args.test_samples}]")

    # 2. Load Model
    print(f"🏗️ Loading Model from {args.checkpoint}...")
    model = Transformer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        heads=args.heads,
        blocks=args.layers,
        dropout=0.0,  # Eval mode không cần dropout
        ff_expansion=args.ff_expansion,
        device=args.device,
    ).to(args.device)

    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)

    # 3. Running Evaluation
    sources = []
    references = []
    candidates = []
    gemini_scores = []

    print(f"🚀 Starting Evaluation with Beam Size = {args.beam_size}...")

    for item in tqdm(dataset):
        src_text = item["translation"]["en"]
        tgt_text = item["translation"]["vi"]

        # Tokenize Source
        # (Lưu ý: Không cần padding batch ở đây vì ta decode từng câu một cho chính xác)
        src_ids = [sp.bos_id()] + sp.encode_as_ids(src_text) + [sp.eos_id()]  # type: ignore
        src_tensor = (
            torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(args.device)
        )

        # --- BEAM SEARCH ---
        pred_ids = beam_search_decode(
            model,
            src_tensor,
            sp,
            args.device,
            beam_size=args.beam_size,
            max_len=args.max_len,
        )

        # Decode về text
        # Lọc bỏ special tokens
        pred_text = sp.decode(pred_ids)  # type: ignore

        sources.append(src_text)
        references.append(tgt_text)
        candidates.append(pred_text)

        # Chấm điểm Gemini (Optional - tốn tiền/quota)
        # Chỉ chấm 10 câu đầu để demo
        if len(gemini_scores) < 10:
            g_score = get_gemini_score(src_text, tgt_text, pred_text)
            if g_score is not None:
                gemini_scores.append(g_score)

    # 4. Compute BLEU
    # SacreBLEU expects references as a list of lists: [[ref1_doc], [ref2_doc]...]
    print("Computing BLEU...")
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    print(f"✅ BLEU Score: {bleu.score:.2f}")

    if gemini_scores:
        avg_gemini = sum(gemini_scores) / len(gemini_scores)
        print(f"🤖 Avg Gemini Score (First 10 samples): {avg_gemini:.2f}/100")

    # 5. Visualization & Analysis
    print("📊 Plotting Metrics...")

    # 5.1 In thử vài mẫu
    print("\n--- SAMPLE TRANSLATIONS ---")
    for i in range(5):
        print(f"Src: {sources[i]}")
        print(f"Ref: {references[i]}")
        print(f"Pred: {candidates[i]}")
        print("-" * 30)

    # 5.2 Plot BLEU analysis (Giả lập Loss vì Loss Inference không quan trọng bằng BLEU)
    # Ta sẽ vẽ "Độ dài câu vs BLEU" - để xem model dịch câu dài hay ngắn tốt hơn

    sent_lens = [len(ref.split()) for ref in references]
    # Chia bin độ dài: 0-10, 10-20, 20-30...
    bins = {}
    for i, length in enumerate(sent_lens):
        bin_idx = (length // 10) * 10
        if bin_idx not in bins:
            bins[bin_idx] = {"refs": [], "cands": []}
        bins[bin_idx]["refs"].append(references[i])
        bins[bin_idx]["cands"].append(candidates[i])

    sorted_bins = sorted(bins.keys())
    bleu_per_bin = []

    for b in sorted_bins:
        if not bins[b]["refs"]:
            bleu_per_bin.append(0)
            continue
        # Tính BLEU cho từng nhóm độ dài
        score = sacrebleu.corpus_bleu(bins[b]["cands"], [bins[b]["refs"]]).score
        bleu_per_bin.append(score)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.bar([f"{b}-{b + 10}" for b in sorted_bins], bleu_per_bin, color="skyblue")
    plt.xlabel("Sentence Length (words)")
    plt.ylabel("BLEU Score")
    plt.title("Translation Quality vs Sentence Length")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Lưu ảnh
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/bleu_analysis.png")
    print("✅ Saved plot to results/bleu_analysis.png")


if __name__ == "__main__":
    main()
