"""Free local judge: the paper's BFCL judge prompt run on an open model on the Colab GPU.

No API key or cost. Prompt and decision rule are copied verbatim from the paper's
scorer (LLMJudgeScorer); only the judge model differs (paper: GPT-4o).
Judge noise is measured by sampling the judge K times at temperature 1.0,
matching the paper's GPT-4o call, which sets no temperature (API default 1.0).

  python src/local_judge.py --runs runs --gt ../ToCall_or_NotToCall/data/BFCL.csv --k 3
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from io_utils import load_jsonl, load_ground_truth, question_of
from free_judge import _JUDGE_SYSTEM, _SCHEMA_HINT, _parse

CONDITIONS = ["no_tool", "force_google", "force_brave"]


def load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant,
                                                 device_map="auto").eval()
    return tok, model


def judge_batch(tok, model, items, temperature, seed, batch_size=16):
    import torch
    torch.manual_seed(seed)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": _JUDGE_SYSTEM + _SCHEMA_HINT},
         {"role": "user", "content": f"QUESTION:\n{q}\n\nREFERENCE:\n{gt}\n\nRESPONSE:\n{resp}"}],
        tokenize=False, add_generation_prompt=True) for q, gt, resp in items]
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=80, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else None,
                                 top_p=1.0, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [_parse(t) for t in texts]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="data/judgments.jsonl")
    a = ap.parse_args()

    gt = load_ground_truth(a.gt)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["condition"], r["question"], r["rep"], r["judge_model"]) for r in load_jsonl(out)}

    tok, model = load_model(a.model)
    with open(out, "a") as f:
        for cond in CONDITIONS:
            path = Path(a.runs) / f"{cond}.jsonl"
            if not path.exists():
                continue
            rows = [r for r in load_jsonl(path) if question_of(r) in gt]
            for rep in range(a.k):
                todo = [r for r in rows if (cond, question_of(r), rep, a.model) not in done]
                if not todo:
                    continue
                items = [(question_of(r), gt[question_of(r)], r["response"]) for r in todo]
                scores = judge_batch(tok, model, items, a.temperature, seed=1000 + rep)
                for r, s in zip(todo, scores):
                    f.write(json.dumps({"condition": cond, "question": question_of(r), "rep": rep,
                                        "judge_model": a.model, "score": s}) + "\n")
                f.flush()
                print(f"{cond} rep {rep}: judged {len(todo)}")
    print(f"judgments cached in {out}")


if __name__ == "__main__":
    main()
