import json
import math
import os
import pandas as pd
import re
import string
import sqlite3
from io import StringIO

from collections import defaultdict
from typing import List

import evaluate

import json
import pandas as pd
import re
from collections import defaultdict
from typing import List
import numpy as np
import ast
import sacrebleu


def get_accuracy_wtq(eval_output, path=None, input_df=None):
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])
    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    def normalize_prediction(pred_str):
        # 1. Try to parse "['value']" into a real list ['value']
        try:
            # Check if it looks like a list
            if pred_str.strip().startswith('[') and pred_str.strip().endswith(']'):
                parsed = ast.literal_eval(pred_str)
                # If it's a list, join it back into a string like "val1, val2"
                if isinstance(parsed, list):
                    return delimiter.join([str(x) for x in parsed])
        except (ValueError, SyntaxError):
            pass
            
        # 2. Fallback: simple character cleanup if ast fails
        # Remove brackets and single quotes
        cleaned = pred_str.replace("[", "").replace("]", "").replace("'", "")
        return cleaned

    delimiter = ", "
    def evaluate_example(predict_str: str, ground_str: str):

        predict_str = normalize_prediction(predict_str)
        ground_str = normalize_prediction(ground_str)
        predict_spans = predict_str.split(delimiter)
        ground_spans = ground_str.split(delimiter)
        predict_values = defaultdict(lambda: 0)
        ground_values = defaultdict(lambda: 0)
        for span in predict_spans:
            span = span.strip()
            try:
                predict_values[float(span.replace(',', ''))] += 1
            except ValueError:
                predict_values[span] += 1
        for span in ground_spans:
            span = span.strip()
            try:
                ground_values[float(span.replace(',', ''))] += 1
            except ValueError:
                ground_values[span] += 1
        _is_correct = predict_values == ground_values
        return _is_correct

    def get_denotation_accuracy(predictions: List[str], references: List[str]):
        assert len(predictions) == len(references)
        correct_num = 0
        for predict_str, ground_str in zip(predictions, references):
            is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
            if is_correct:
                correct_num += 1
        return correct_num / len(predictions)
    
    acc = get_denotation_accuracy(df["pred"].tolist(), df["label"].tolist())

    return acc


def get_accuracy_tabfact(eval_output, path=None, input_df=None):
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])
    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    delimiter = ", "
    def evaluate_example(predict_str: str, ground_str: str):
        predict_str = predict_str.split('.')[0]

        predict_spans = predict_str.split(delimiter)
        ground_spans = ground_str.split(delimiter)
        predict_values = defaultdict(lambda: 0)
        ground_values = defaultdict(lambda: 0)
        for span in predict_spans:
            span = span.strip()
            try:
                predict_values[float(span.replace(',', ''))] += 1
            except ValueError:
                predict_values[span] += 1
        for span in ground_spans:
            span = span.strip()
            try:
                ground_values[float(span.replace(',', ''))] += 1
            except ValueError:
                ground_values[span] += 1
        _is_correct = predict_values == ground_values
        return _is_correct

    def get_denotation_accuracy(predictions: List[str], references: List[str]):
        assert len(predictions) == len(references)
        correct_num = 0
        for predict_str, ground_str in zip(predictions, references):
            is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
            if is_correct:
                correct_num += 1
        return correct_num / len(predictions)

    acc = get_denotation_accuracy(df["pred"].tolist(), df["label"].tolist())

    return acc


def get_accuracy_hctqa(eval_output, path=None, input_df=None):
    """
    Evaluate HCTQA predictions using F1 Score and Complete Containment (CC Score).

    Answer format: "{val1 | val2} || {val3 | val4}"
    Each {…} is one row result, || separates multiple rows,
    | separates values within a row.

    Metrics (from the HCT-QA paper):
      - F1 Score: Token-level F1 between predicted and gold values.
        Rewards partial correctness.
      - CC Score: Binary score, 1 iff recall == 1.0 (the prediction
        contains the entire ground truth), else 0.

    Returns:
      dict with keys: "f1" (avg F1), "cc" (avg CC score)
    """
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    def _normalize_value(v):
        """Normalize a single value: lowercase, strip quotes, handle numeric commas."""
        v = v.strip().lower()
        # Remove surrounding quotes
        v = v.strip("'\"")
        # Strip commas from numbers (e.g. "1,157" -> "1157")
        v_no_comma = v.replace(',', '')
        try:
            import math
            num = float(v_no_comma)
            if not math.isfinite(num):
                return v
            # Normalize to consistent string representation
            # Use int representation if it's a whole number
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return v

    def _parse_hctqa_answer(answer_str):
        """
        Parse '{a | b} || {c | d}' into a flat list of normalized values.

        """
        answer_str = str(answer_str).strip()
        if not answer_str or answer_str.lower() in ('nan', 'none', 'no answer', ''):
            return []

        # Split on || (row separator)
        row_parts = answer_str.split('||')
        values = []
        for part in row_parts:
            part = part.strip().strip('{}')
            if not part:
                continue
            # Split on | (value separator within a row)
            for v in part.split('|'):
                normalized = _normalize_value(v)
                if normalized:  # skip empty strings
                    values.append(normalized)
        return values

    def _compute_f1_and_cc(pred_str, gold_str):
        """
        Compute token-level Precision, Recall, F1, and CC Score
        for a single (prediction, gold) pair.

        Uses multiset (bag) matching: each predicted value can match
        at most one gold value and vice versa.

        Returns: (precision, recall, f1, cc)
        """
        pred_values = _parse_hctqa_answer(pred_str)
        gold_values = _parse_hctqa_answer(gold_str)

        # Edge cases
        if len(gold_values) == 0 and len(pred_values) == 0:
            return 1.0, 1.0, 1.0, 1
        if len(gold_values) == 0:
            return 0.0, 1.0, 0.0, 1  # nothing to recall
        if len(pred_values) == 0:
            return 0.0, 0.0, 0.0, 0  # missed everything

        # Multiset matching: greedily match pred values to gold values
        gold_remaining = list(gold_values)
        num_matched = 0

        for pv in pred_values:
            if pv in gold_remaining:
                gold_remaining.remove(pv)
                num_matched += 1

        precision = num_matched / len(pred_values) if len(pred_values) > 0 else 0.0
        recall = num_matched / len(gold_values) if len(gold_values) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # CC Score: 1 iff recall == 1.0 (prediction fully contains ground truth)
        cc = 1 if recall == 1.0 else 0

        return precision, recall, f1, cc

    total = len(df)
    sum_f1 = 0.0
    sum_cc = 0.0

    for _, row in df.iterrows():
        pred_str = str(row["pred"])
        gold_str = str(row["label"])
        _, _, f1, cc = _compute_f1_and_cc(pred_str, gold_str)
        sum_f1 += f1
        sum_cc += cc

    avg_f1 = sum_f1 / total if total > 0 else 0.0
    avg_cc = sum_cc / total if total > 0 else 0.0

    return {"f1": avg_f1, "cc": avg_cc}


def get_accuracy_hctqa_stress_test(eval_output, path=None, input_df=None):
    """Like get_accuracy_hctqa but also reports per-template breakdown."""
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    result = get_accuracy_hctqa(None, input_df=df)
    avg_f1, avg_cc = result["f1"], result["cc"]
    print(f"Overall ({len(df)} samples): F1={avg_f1:.4f}  CC={avg_cc:.4f}")

    if "question_type" in df.columns:
        for template, group in df.groupby("question_type"):
            sub = get_accuracy_hctqa(None, input_df=group.reset_index(drop=True))
            print(f"  {template} ({len(group)} samples): F1={sub['f1']:.4f}  CC={sub['cc']:.4f}")
            result[template] = sub

    return result


def get_accuracy_scitat(eval_output, path=None, input_df=None):
    """EM for short-form answers (all gold spans ≤5 words), token-F1 + BERTScore for free-form.

    Follows the official SciTab evaluation protocol:
      - label is stored as a JSON list of spans (one element per sub-answer)
      - short-form: all spans have ≤5 words → Exact Match  (official: d["answer_en"] list check)
      - free-form:  any span >5 words → token-level F1 and BERTScore F1
      - gold spans are joined with ". " before metric computation  (official: ". ".join(d["answer"]))
    """
    from bert_score import score as bert_score_fn

    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    def _parse_label(s: str) -> list:
        """Recover the original list of spans from a JSON-encoded label."""
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [s.strip()]

    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def _is_short_form(gold_spans: list) -> bool:
        """Mirrors: all([len(a.split(" ")) <= 5 for a in d["answer_en"]])"""
        return all(len(s.split(" ")) <= 5 for s in gold_spans)

    def _exact_match(pred: str, gold: str) -> float:
        return float(_normalize(pred) == _normalize(gold))

    def _token_f1(pred: str, gold: str) -> float:
        pred_tokens = _normalize(pred).split()
        gold_tokens = _normalize(gold).split()
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        pred_counts = defaultdict(int)
        gold_counts = defaultdict(int)
        for t in pred_tokens:
            pred_counts[t] += 1
        for t in gold_tokens:
            gold_counts[t] += 1
        overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
        precision = overlap / len(pred_tokens)
        recall    = overlap / len(gold_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    preds  = [str(r) for r in df["pred"].tolist()]
    labels = [str(r) for r in df["label"].tolist()]

    em_scores, f1_scores = [], []
    short_mask, free_mask = [], []
    gold_strs = []

    for pred, label in zip(preds, labels):
        gold_spans = _parse_label(label)
        gold_str   = '. '.join(gold_spans)   # mirrors ". ".join(d["answer"]) in official eval
        gold_strs.append(gold_str)

        if _is_short_form(gold_spans):
            em_scores.append(_exact_match(pred, gold_str))
            short_mask.append(True)
            free_mask.append(False)
        else:
            f1_scores.append(_token_f1(pred, gold_str))
            short_mask.append(False)
            free_mask.append(True)

    n_short = sum(short_mask)
    n_free  = sum(free_mask)

    em = np.mean(em_scores) if em_scores else float('nan')
    f1 = np.mean(f1_scores) if f1_scores else float('nan')

    # BERTScore on free-form subset — gold joined with ". " (mirrors official ". ".join(d["answer"]))
    free_preds  = [p for p, m in zip(preds,     free_mask) if m]
    free_labels = [g for g, m in zip(gold_strs, free_mask) if m]
    if free_preds:
        local_bert = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "google-bert", "bert-base-uncased",
        )
        bert_model = local_bert if os.path.isdir(local_bert) else "bert-base-uncased"
        _, _, F1 = bert_score_fn(
            free_preds, free_labels,
            model_type=bert_model,
            num_layers=12,
            lang="en",
            verbose=False,
        )
        bertscore = F1.mean().item()
    else:
        bertscore = float('nan')

    result = {
        "em":           round(em, 4),
        "f1":           round(f1, 4),
        "bertscore_f1": round(bertscore, 4),
        "n_short_form": n_short,
        "n_free_form":  n_free,
    }

    print(f"  Short-form ({n_short} samples): EM  = {em:.4f}")
    print(f"  Free-form  ({n_free} samples): F1  = {f1:.4f}")
    print(f"  Free-form  ({n_free} samples): BERTScore F1 = {bertscore:.4f}")

    return result


def get_accuracy_tqa_bench(eval_output, path=None, input_df=None):
    """Accuracy for TQA-Bench (multiple-choice A/B/C/D), broken down by qtype and domain."""
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + '\n')

    def extract_option(pred: str) -> str:
        pred = pred.strip()
        if pred.upper() in ('A', 'B', 'C', 'D'):
            return pred.upper()
        m = re.search(r'\b([ABCD])[.):\s]', pred.upper())
        if m:
            return m.group(1)
        for ch in pred.upper():
            if ch in ('A', 'B', 'C', 'D'):
                return ch
        return pred.upper()[:1] if pred else ''

    preds  = [extract_option(str(p)) for p in df['pred'].tolist()]
    labels = [str(l).strip().upper() for l in df['label'].tolist()]

    correct = sum(p == l for p, l in zip(preds, labels))
    overall_acc = correct / len(preds) if preds else 0.0
    result = {'overall_acc': overall_acc}

    for col, heading in [('question_type', 'Question Type'), ('domain', 'Domain')]:
        if col not in df.columns:
            continue
        print(f"\n{heading:<30} {'Acc':>8} {'Count':>6}")
        print('-' * 48)
        for key, group in sorted(df.groupby(col)):
            g_preds  = [extract_option(str(p)) for p in group['pred'].tolist()]
            g_labels = [str(l).strip().upper() for l in group['label'].tolist()]
            acc = sum(p == l for p, l in zip(g_preds, g_labels)) / len(g_preds)
            result[f'{col}:{key}'] = {'acc': acc, 'count': len(group)}
            print(f'{key:<30} {acc:>8.4f} {len(group):>6}')
        print('-' * 48)
        print(f"{'Overall':<30} {overall_acc:>8.4f} {len(df):>6}")

    return result


def _normalize_cell(v):
    v = str(v).strip().lower().strip("'\"")
    try:
        num = float(v.replace(',', ''))
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, AttributeError):
        return v


def _parse_linearized_table(text):
    """Parse 'col : h1 | h2 row 1 : v1 | v2 ...' into {'columns': [...], 'rows': [...]}."""
    text = str(text).strip()

    col_match = re.match(r'col\s*:\s*(.*?)(?=\s+row\s+\d|\Z)', text, re.IGNORECASE | re.DOTALL)
    if not col_match:
        return {'columns': [], 'rows': []}

    columns = [_normalize_cell(h) for h in col_match.group(1).strip().split('|')]
    row_matches = re.findall(
        r'row\s+\d+\s*:\s*(.*?)(?=\s+row\s+\d|\Z)', text, re.IGNORECASE | re.DOTALL
    )
    rows = [[_normalize_cell(v) for v in r.split('|')] for r in row_matches]
    return {'columns': columns, 'rows': rows}


def get_table_metrics(eval_output, path=None, input_df=None):
    """Table EM, Row EM, Column EM, and Cell EM for result-table generation tasks.

    Table EM:  exact match of the full table (headers + ordered rows).
    Row EM:    precision/recall treating rows as an unordered set
               (values within each row are ordered).
    Column EM: precision/recall matching columns by header and ordered values.
    Cell EM:   precision/recall treating individual cell values as an unordered set.

    Precision and recall are computed globally across the evaluation set.
    """
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + '\n')

    table_em_count = 0
    row_pred_total = row_gold_total = row_matched = 0
    col_pred_total = col_gold_total = col_matched = 0
    cell_pred_total = cell_gold_total = cell_matched = 0
    total = len(df)

    for _, row in df.iterrows():
        pred_t = _parse_linearized_table(str(row['pred']))
        gold_t = _parse_linearized_table(str(row['label']))

        # Table EM: headers match AND all rows match in order
        if (pred_t['columns'] == gold_t['columns']
                and len(pred_t['rows']) == len(gold_t['rows'])
                and all(pr == gr for pr, gr in zip(pred_t['rows'], gold_t['rows']))):
            table_em_count += 1

        # Row EM: unordered matching of rows (values within row ordered)
        pred_rows = [tuple(r) for r in pred_t['rows']]
        gold_rows = [tuple(r) for r in gold_t['rows']]
        gold_rem = list(gold_rows)
        rm = 0
        for pr in pred_rows:
            if pr in gold_rem:
                gold_rem.remove(pr)
                rm += 1
        row_matched += rm
        row_pred_total += len(pred_rows)
        row_gold_total += len(gold_rows)

        # Column EM: match by (header, ordered column values)
        def _cols_as_pairs(t):
            return [(h, tuple(r[i] if i < len(r) else '' for r in t['rows']))
                    for i, h in enumerate(t['columns'])]

        pred_cols = _cols_as_pairs(pred_t)
        gold_cols = _cols_as_pairs(gold_t)
        gold_cols_rem = list(gold_cols)
        cm = 0
        for pc in pred_cols:
            if pc in gold_cols_rem:
                gold_cols_rem.remove(pc)
                cm += 1
        col_matched += cm
        col_pred_total += len(pred_cols)
        col_gold_total += len(gold_cols)

        # Cell EM: unordered multiset of all cell values
        pred_cells = [v for r in pred_t['rows'] for v in r]
        gold_cells = [v for r in gold_t['rows'] for v in r]
        gold_cells_rem = list(gold_cells)
        celm = 0
        for pc in pred_cells:
            if pc in gold_cells_rem:
                gold_cells_rem.remove(pc)
                celm += 1
        cell_matched += celm
        cell_pred_total += len(pred_cells)
        cell_gold_total += len(gold_cells)

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    table_em = table_em_count / total if total > 0 else 0.0
    row_p = row_matched / row_pred_total if row_pred_total else 0.0
    row_r = row_matched / row_gold_total if row_gold_total else 0.0
    row_f1 = _f1(row_p, row_r)
    col_p = col_matched / col_pred_total if col_pred_total else 0.0
    col_r = col_matched / col_gold_total if col_gold_total else 0.0
    col_f1 = _f1(col_p, col_r)
    cell_p = cell_matched / cell_pred_total if cell_pred_total else 0.0
    cell_r = cell_matched / cell_gold_total if cell_gold_total else 0.0
    cell_f1 = _f1(cell_p, cell_r)

    print(f"  Table EM:   {table_em:.4f}")
    print(f"  Row EM:     P={row_p:.4f}  R={row_r:.4f}  F1={row_f1:.4f}")
    print(f"  Column EM:  P={col_p:.4f}  R={col_r:.4f}  F1={col_f1:.4f}")
    print(f"  Cell EM:    P={cell_p:.4f}  R={cell_r:.4f}  F1={cell_f1:.4f}")

    return {
        'table_em':          round(table_em, 4),
        'row_em_precision':  round(row_p, 4),
        'row_em_recall':     round(row_r, 4),
        'row_em_f1':         round(row_f1, 4),
        'col_em_precision':  round(col_p, 4),
        'col_em_recall':     round(col_r, 4),
        'col_em_f1':         round(col_f1, 4),
        'cell_em_precision': round(cell_p, 4),
        'cell_em_recall':    round(cell_r, 4),
        'cell_em_f1':        round(cell_f1, 4),
    }



def get_accuracy_spider_text2sql(eval_output, path=None, input_df=None):
    """Execution accuracy for Spider text-to-SQL.

    Each `label` is a JSON string produced by SpiderText2SQLDataset containing:
      - 'gold_sql':    the reference SQL query
      - 'answer':      pre-computed gold result (JSON orient='split')
      - 'tables':      list of table JSON strings (orient='split')
      - 'table_names': list of table name strings

    The predicted SQL (`pred`) is executed against an in-memory SQLite DB
    reconstructed from the stored table data and compared to the gold result.
    """
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + '\n')

    def _build_conn(tables_json, table_names):
        conn = sqlite3.connect(':memory:')
        conn.text_factory = lambda b: b.decode(errors='replace')
        for t_json, name in zip(tables_json, table_names):
            df_t = pd.read_json(StringIO(t_json), orient='split')
            df_t.to_sql(name, conn, index=False, if_exists='replace')
        return conn

    def _normalize_result(rows):
        return sorted(tuple(str(v).strip().lower() for v in row) for row in rows)

    def _extract_sql(text):
        """Strip markdown code fences if the model wrapped the SQL."""
        text = text.strip()
        m = re.search(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text

    ex_scores = []
    for _, row in df.iterrows():
        pred_sql = _extract_sql(str(row['pred']))
        try:
            meta = json.loads(str(row['eval_meta']))
            gold_result = json.loads(meta['answer'])['data']
            conn = _build_conn(meta['tables'], meta['table_names'])
            cur = conn.cursor()
            cur.execute(pred_sql)
            pred_result = cur.fetchall()
            conn.close()
            match = int(_normalize_result(gold_result) == _normalize_result(pred_result))
        except Exception:
            match = 0
        ex_scores.append(match)

    ex_acc = sum(ex_scores) / len(ex_scores) if ex_scores else 0.0
    print(f"  Execution Accuracy: {ex_acc:.4f} ({sum(ex_scores)}/{len(ex_scores)})")

    df['ex'] = ex_scores
    if path:
        df.to_csv(path.replace('.csv', '_ex_details.csv'), index=False)

    return ex_acc


def _pipe_bag_em(eval_output, path, input_df, metric_name):
    """Shared bag-of-values EM for datasets using ' | ' as multi-value delimiter.

    Both label and prediction are split on ' | '. Each value is normalized:
      - Numbers: strip commas, parse as float, canonicalize whole-number floats
        to int strings ('460.0' -> '460').
      - Strings: lowercase, strip leading/trailing whitespace.
    Comparison is unordered (multiset), so value order does not matter.
    """
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])

    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    def _normalize(v):
        v = str(v).strip().lower()
        v = v.lstrip('$€£¥')
        try:
            num = float(v.replace(',', ''))
            if not (num == num) or num == float('inf') or num == float('-inf'):
                return v
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return v

    def _to_bag(s):
        bag = defaultdict(int)
        for span in str(s).split(' | '):
            bag[_normalize(span)] += 1
        return bag

    preds  = [str(r) for r in df["pred"].tolist()]
    labels = [str(r) for r in df["label"].tolist()]

    scores = [float(_to_bag(p) == _to_bag(l)) for p, l in zip(preds, labels)]
    acc = sum(scores) / len(scores) if scores else 0.0
    print(f"  {metric_name}: {acc:.4f}  ({len(scores)} samples)")
    return acc


def get_accuracy_mmqa(eval_output, path=None, input_df=None):
    return _pipe_bag_em(eval_output, path, input_df, metric_name="Accuracy")


def get_accuracy_multihiertt(eval_output, path=None, input_df=None):
    em = _pipe_bag_em(eval_output, path, input_df, metric_name="Exact Match")
    return {"em": round(em, 4)}


eval_funcs = {
    'wtq' : get_accuracy_wtq,
    'wikisql' : get_accuracy_wtq,
    'structProbe': get_accuracy_wtq,
    'hitab': get_accuracy_wtq,
    'tabfact': get_accuracy_tabfact,
    'hctqa': get_accuracy_hctqa,
    'hctqa_stress_test': get_accuracy_hctqa_stress_test,
    'mmqa': get_accuracy_mmqa,
    'multihiertt': get_accuracy_multihiertt,
    'tabmwp': get_accuracy_wtq,
    'scitat': get_accuracy_scitat,
    'tqa_bench': get_accuracy_tqa_bench,
    'atis': get_table_metrics,
    'geoquery': get_table_metrics,
    'spider_sql': get_table_metrics,
    'spider_text2sql': get_accuracy_spider_text2sql,
}
