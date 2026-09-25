"""
Business Entity Resolution — Name Blocking (Person 2's slice)
Fixes vs. the previous version:
  1. S1 (reference) may be sampled — that's just "how many S1 records to test".
  2. S2 / S3 must be loaded in FULL — a true match could be anywhere in
     millions of rows, so truncating them silently kills recall (this is
     why we saw ~0.01-0.02% recall last run).
  3. Correct schema per README: entity_id, business_name, business_address,
     country — tab-separated (.tsv), NOT csv. sep="\t" is mandatory.
  4. Ground truth format: source1_entity_id, matched_entity_ids
     where matched_entity_ids is a SINGLE comma-separated list mixing
     S2- and S3- prefixed ids together. We split it and bucket by prefix
     to score S1xS2 and S1xS3 recall separately.
"""

import re
import time
import gc
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\hp\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
S1_SAMPLE_ROWS = None       # FULL RUN: was 50_000 (dev sample) -> None (all ~2.2M S1 records)
S2_SAMPLE_ROWS = None       # MUST be None (full load) for recall to mean anything
S3_SAMPLE_ROWS = None       # MUST be None (full load) for recall to mean anything
TOP_K_NGRAM = 20            # candidates per S1 record from char-ngram blocking

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_data():
    t0 = time.time()
    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t", nrows=S1_SAMPLE_ROWS)
    s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t", nrows=S2_SAMPLE_ROWS)
    s3 = pd.read_csv(f"{DATA_DIR}/train_source3.tsv", sep="\t", nrows=S3_SAMPLE_ROWS)
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t")
    print(f"  s1={len(s1)} s2={len(s2)} s3={len(s3)} gt={len(gt)}  ({time.time()-t0:.1f}s)")
    return s1, s2, s3, gt


def build_ground_truth_sets(gt):
    """
    gt has: source1_entity_id, matched_entity_ids (comma-separated, mixed S2/S3, may be NaN/empty)
    Returns two dicts: s1_id -> set(matched S2 ids), s1_id -> set(matched S3 ids)
    """
    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        raw = row.matched_entity_ids
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        for mid in str(raw).split(","):
            mid = mid.strip()
            if not mid:
                continue
            if mid.startswith("S2-"):
                gt_s2[s1_id].add(mid)
            elif mid.startswith("S3-"):
                gt_s3[s1_id].add(mid)
    return gt_s2, gt_s3

# ---------------------------------------------------------------------------
# NAME NORMALIZATION
# ---------------------------------------------------------------------------
import unicodedata

# ---------------------------------------------------------------------------
# Name normalization -- ported from Vid's src/normalization.py so blocking
# uses the same logic the team will standardize on, PLUS the extra legal
# forms her exploration_report.md found that aren't in her file yet
# (sarl/sas/sasu/eurl/pllc/pc/lp/llp) -- flagged back to her to fold in.
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """'Café' -> 'Cafe'. Algorithmic (stdlib), not a transliteration lookup --
    does NOT convert non-Latin scripts (Hindi/Bengali/Tamil/etc.), which is
    intentional per the team's principle of not inventing translations."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_punct_keep_alnum(text):
    return re.sub(r"[^\w\s]", " ", text)


NAME_ABBREVIATIONS = {
    "corp": "corporation", "co": "company", "inc": "incorporated",
    "ltd": "limited", "llc": "llc", "pvt": "private", "pte": "private",
    "intl": "international", "mfg": "manufacturing", "assn": "association",
    "assoc": "association", "bros": "brothers", "grp": "group",
    "svcs": "services", "svc": "service", "dept": "department", "&": "and",
}

LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "llp", "pvt", "private", "pte", "gmbh", "sa", "srl",
    "plc",
    # added from EDA (not yet in Vid's file as of the version she shared):
    "sarl", "sas", "sasu", "eurl", "pllc", "pc", "lp",
    # added from zero-candidate diagnosis (Hernandez Regional Aktiengesellschaft):
    "aktiengesellschaft", "ag",
}

# Collapses dotted abbreviations (L.L.C, P.C, S.A.) into a joined token
# BEFORE punctuation stripping, so they can match LEGAL_SUFFIX_TOKENS above.
DOTTED_ABBREV_RE = re.compile(r"\b(?:[a-zA-Z]\.){2,}")


def normalize_name(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    text = str(name)
    text = DOTTED_ABBREV_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    text = strip_accents(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = strip_punct_keep_alnum(text)
    tokens = text.split()
    expanded = [NAME_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    return collapse_whitespace(" ".join(expanded))


def core_name(name: str) -> str:
    """
    normalize_name() with legal-suffix tokens removed from ANYWHERE in the
    name, not just the trailing position.

    Originally trailing-only (matching Vid's normalization.py). Diagnosis of
    zero-candidate S1 records showed the data reorders names (e.g. "Inc Knox
    and Nicholas" instead of "Knox and Nicholas, Inc") -- when that happens,
    trailing-only stripping leaves "inc" on one side and not the other, so
    the two records end up with different token SETS entirely. That breaks
    not just exact match but sorted_token_block too, since sorting only
    fixes order, not set membership. Stripping suffix tokens wherever they
    appear keeps both sides consistent regardless of word order.
    """
    normalized = normalize_name(name)
    tokens = [t for t in normalized.split() if t not in LEGAL_SUFFIX_TOKENS]
    return " ".join(tokens)

# ---------------------------------------------------------------------------
# BLOCKING METHODS
# ---------------------------------------------------------------------------
def exact_block(s1_norm, target_df, target_norm_col):
    """Exact normalized-name match. Returns dict s1_id -> set(target_ids)."""
    index = defaultdict(list)
    for eid, name in zip(target_df["entity_id"], target_df[target_norm_col]):
        if name:
            index[name].append(eid)
    result = {}
    for eid, name in s1_norm.items():
        result[eid] = set(index.get(name, []))
    return result


def sorted_token_block(s1_norm, target_df, target_norm_col):
    """Block on sorted-token key (handles word reordering)."""
    def token_key(name):
        return " ".join(sorted(name.split()))

    index = defaultdict(list)
    for eid, name in zip(target_df["entity_id"], target_df[target_norm_col]):
        if name:
            index[token_key(name)].append(eid)
    result = {}
    for eid, name in s1_norm.items():
        result[eid] = set(index.get(token_key(name), []))
    return result


def char_ngram_block(s1_norm, target_df, target_norm_col, n=3, top_k=TOP_K_NGRAM,
                      max_df_abs=1000, rarest_terms_per_query=10,
                      max_candidates_per_query=3000):
    """
    TF-IDF char n-gram cosine similarity blocking (catches typos/reordering),
    scaled for millions of target rows via an inverted index.

    Two earlier attempts failed at this scale:
      - brute-force NearestNeighbors: compares every query against every
        target (50k x 5M+ = tens of billions of ops).
      - naive inverted index with max_df as a FRACTION (0.3): on 5M rows,
        a trigram present in 30% of names still has 1.5M-row postings, so
        unioning postings for all of a query's trigrams re-creates a
        near-brute-force candidate pool per query.

    Fix: (1) drop trigrams whose ABSOLUTE document frequency exceeds
    max_df_abs (so no single term can ever contribute a huge postings
    list, regardless of corpus size), and (2) for each query, only use its
    `rarest_terms_per_query` trigrams (the most discriminative ones) to
    build the candidate pool, instead of all of them.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    target_names = target_df[target_norm_col].fillna("").tolist()
    target_ids = target_df["entity_id"].tolist()

    # NOTE: max_df is intentionally NOT passed to TfidfVectorizer here.
    # TfidfVectorizer's max_df doesn't cap postings length -- it deletes the
    # term from the vocabulary entirely for any trigram above that document
    # frequency. On this corpus that wiped out common-but-still-meaningful
    # trigrams ("ing", "ati", "com"...), leaving many S1 queries with an
    # all-zero vector -> zero candidates (this is why zero_candidates jumped
    # from ~400k to 1.85M). The postings-size protection instead happens
    # below, per-query, via rarest_terms_per_query + the max_df_abs mask.
    # dtype=float32 (default is float64): halves the memory footprint of the
    # sparse matrix and its CSC copy below -- this was the difference
    # between fitting in RAM and an ArrayMemoryError on the tocsc() copy.
    print(f"    char_ngram: fitting vectorizer on {len(target_names):,} target docs...")
    _t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(n, n),
                                  min_df=1, dtype=np.float32)
    target_vecs = vectorizer.fit_transform(target_names).tocsr()
    print(f"    char_ngram: fit done, vocab={len(vectorizer.vocabulary_):,} ({time.time()-_t0:.1f}s)")
    _t0 = time.time()
    target_vecs_csc = target_vecs.tocsc()
    print(f"    char_ngram: csc conversion done ({time.time()-_t0:.1f}s)")

    doc_freq = np.diff(target_vecs_csc.indptr)  # postings length per term, vectorized

    s1_ids = list(s1_norm.keys())
    s1_names = list(s1_norm.values())
    s1_vecs = vectorizer.transform(s1_names).tocsr()

    result = {}
    _t0 = time.time()
    _n = len(s1_ids)
    for row_i, eid in enumerate(s1_ids):
        if row_i and row_i % 2_000 == 0:
            _rate = row_i / (time.time() - _t0)
            _eta_min = (_n - row_i) / _rate / 60
            print(f"    char_ngram: {row_i:,}/{_n:,} ({_rate:.0f} rows/s, ETA {_eta_min:.1f} min)")
        query_terms = s1_vecs.indices[s1_vecs.indptr[row_i]:s1_vecs.indptr[row_i + 1]]
        if len(query_terms) == 0:
            result[eid] = set()
            continue

        # keep only the rarest terms present in this query -- these are the
        # most discriminative and keep the candidate pool small by construction
        if len(query_terms) > rarest_terms_per_query:
            term_dfs = doc_freq[query_terms]
            keep = np.argsort(term_dfs)[:rarest_terms_per_query]
            query_terms = query_terms[keep]

        # Safety net for postings-list size (this is what max_df was meant to
        # do): drop any selected term whose target-corpus doc frequency still
        # exceeds max_df_abs.
        safe_mask = doc_freq[query_terms] <= max_df_abs
        if safe_mask.any():
            query_terms = query_terms[safe_mask]
        else:
            # Every term available for this query is common (df > max_df_abs).
            # Falling back to ALL of them unfiltered (previous behavior) could
            # concatenate several million-length postings lists for a single
            # query -- exactly the pathological case max_df_abs exists to
            # prevent. Use just the single least-common one instead: still
            # bounded, still returns something, no explosion.
            single_rarest = query_terms[np.argmin(doc_freq[query_terms])]
            query_terms = np.array([single_rarest])

        postings_list = [
            target_vecs_csc.indices[target_vecs_csc.indptr[j]:target_vecs_csc.indptr[j + 1]]
            for j in query_terms
        ]
        concat = np.concatenate(postings_list) if postings_list else np.array([], dtype=int)
        if len(concat) == 0:
            result[eid] = set()
            continue

        # Rank candidates by how many of the query's rare trigrams they
        # actually share BEFORE capping -- previously we capped by raw row
        # index order, which could silently drop the true match (e.g. typo
        # cases like "Na0n" vs "Naon") whenever it happened to have a high
        # row index and the candidate pool exceeded the cap.
        cand_idx, overlap_counts = np.unique(concat, return_counts=True)
        if len(cand_idx) > max_candidates_per_query:
            top_by_overlap = np.argsort(-overlap_counts)[:max_candidates_per_query]
            cand_idx = cand_idx[top_by_overlap]

        sub = target_vecs[cand_idx]
        q = s1_vecs[row_i]
        sims = sub.dot(q.T).toarray().ravel()

        k = min(top_k, len(cand_idx))
        top_local = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        result[eid] = {target_ids[cand_idx[i]] for i in top_local}

    return result

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(candidates, n_s1, n_target):
    total = sum(len(v) for v in candidates.values())
    zero = sum(1 for v in candidates.values() if len(v) == 0)
    max_c = max((len(v) for v in candidates.values()), default=0)
    bruteforce = n_s1 * n_target
    reduction = 1 - (total / bruteforce) if bruteforce else 0
    return {
        "n_s1": n_s1,
        "avg_candidates_per_s1": round(total / n_s1, 2) if n_s1 else 0,
        "max_candidates": max_c,
        "n_with_zero_candidates": zero,
        "candidate_reduction_vs_bruteforce": f"{reduction*100:.2f}%",
    }


def compute_recall(candidates, gt_dict, tested_s1_ids):
    """
    Recall = fraction of true (S1, target) pairs that landed inside the
    candidate set -- but ONLY over S1 ids we actually sampled and ran
    blocking on. gt_dict is built from the FULL ground truth file
    (2.2M+ rows), which covers far more S1 records than the 50k we
    sampled -- scoring against the full gt_dict would count every
    un-sampled S1 id as a "miss" and massively deflate recall.
    """
    hit, total = 0, 0
    for s1_id in tested_s1_ids:
        true_ids = gt_dict.get(s1_id)
        if not true_ids:
            continue
        cand = candidates.get(s1_id, set())
        for tid in true_ids:
            total += 1
            if tid in cand:
                hit += 1
    return hit / total if total else float("nan")

def diagnose_zero_candidates(s1, target_df, union, gt_dict, label, n_examples=15):
    """
    Print actual S1 name vs. true-match name for records that got ZERO
    candidates, so we can categorize *why* blocking missed them instead of
    guessing (suffix mismatch? typo? multilingual/no Latin overlap? word
    reorder? or just no true match at all, which is valid per the README).
    """
    print(f"\n--- Zero-candidate examples: S1 x {label} ---")
    target_lookup = dict(zip(target_df["entity_id"], target_df["business_name"]))
    s1_name_lookup = dict(zip(s1["entity_id"], s1["business_name"]))

    shown = 0
    for s1_id, cand in union.items():
        if len(cand) > 0:
            continue
        true_ids = gt_dict.get(s1_id)
        if not true_ids:
            continue  # no true match to compare against -- not a real miss
        s1_name = s1_name_lookup.get(s1_id, "?")
        for tid in list(true_ids)[:1]:
            true_name = target_lookup.get(tid, "?")
            print(f"  S1: {s1_name!r:50}  ->  TRUE MATCH ({tid}): {true_name!r}")
        shown += 1
        if shown >= n_examples:
            break
    if shown == 0:
        print("  (no zero-candidate records with a known true match found in this sample)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_pair(s1_norm, target_df, target_norm_col, gt_dict, label, n_s1):
    print(f"\n=== Blocking S1 x {label} ===")
    n_target = len(target_df)

    t0 = time.time()
    exact = exact_block(s1_norm, target_df, target_norm_col)
    print(f"  exact_block done ({time.time()-t0:.1f}s):        {compute_metrics(exact, n_s1, n_target)}")

    t0 = time.time()
    token = sorted_token_block(s1_norm, target_df, target_norm_col)
    print(f"  sorted_token_block done ({time.time()-t0:.1f}s): {compute_metrics(token, n_s1, n_target)}")

    # FUNNEL: only run the expensive char-ngram pass on S1 records that are
    # STILL zero-candidate after exact+token blocking. Measured: char_ngram
    # on all 2.2M rows runs ~580 rows/s (~58 min/source) once every row does
    # real work (no more empty-vector skips). exact+token already cover the
    # other ~83% with >=1 candidate, so running the slow matcher on all of
    # them bought nothing for that 83% and cost ~50 wasted minutes/source.
    #
    # Trade-off (documented, not free): a record with SOME exact/token
    # candidates but not the TRUE one is not zero-candidate, so it skips the
    # ngram pass here and could miss a true match ngram blocking would have
    # caught. Zero-candidate count is the dominant recall lever we've been
    # tracking all along, so this trades a small amount of that long-tail
    # recall for a ~6x runtime cut -- log it as a deliberate choice in the
    # write-up, not a silent behavior change.
    pre_union = {eid: exact.get(eid, set()) | token.get(eid, set()) for eid in s1_norm}
    zero_ids = {eid for eid, cands in pre_union.items() if not cands}
    print(f"  char_ngram: running only on {len(zero_ids):,}/{n_s1:,} still-zero records "
          f"({len(zero_ids)/n_s1*100:.1f}%) instead of all of them")

    # Free the big exact/token candidate dicts (millions of sets of strings)
    # before the heavy TF-IDF step -- they're already folded into pre_union
    # and not needed again until the final union at the bottom.
    del exact, token
    gc.collect()

    s1_norm_subset = {eid: s1_norm[eid] for eid in zero_ids}
    t0 = time.time()
    if s1_norm_subset:
        ngram_subset = char_ngram_block(s1_norm_subset, target_df, target_norm_col)
        ngram_metrics = compute_metrics(ngram_subset, len(s1_norm_subset), n_target)
    else:
        ngram_subset = {}
        ngram_metrics = "n/a (no zero-candidate records)"
    print(f"  char_ngram_block done ({time.time()-t0:.1f}s):   {ngram_metrics}")

    union = {eid: pre_union[eid] | ngram_subset.get(eid, set()) for eid in s1_norm}
    print(f"  union:                                    {compute_metrics(union, n_s1, n_target)}")

    recall = compute_recall(union, gt_dict, s1_norm.keys())
    print(f"\nRECALL against {label}- ground truth: {recall*100:.2f}%")
    return union


def main():
    print("Loading data...")
    s1, s2, s3, gt = load_data()

    s1["name_norm"] = s1["business_name"].apply(core_name)
    s1_norm = dict(zip(s1["entity_id"], s1["name_norm"]))
    gt_s2, gt_s3 = build_ground_truth_sets(gt)

    out_path = "candidate_pairs.tsv"
    with open(out_path, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tblocking_method\n")
    total_written = 0

    # ---- source2: process, write, then FREE before touching source3 ----
    s2["name_norm"] = s2["business_name"].apply(core_name)
    union_s2 = run_pair(s1_norm, s2, "name_norm", gt_s2, "source2", len(s1))
    diagnose_zero_candidates(s1, s2, union_s2, gt_s2, "source2")
    with open(out_path, "a") as f:
        for eid, cands in union_s2.items():
            for tid in cands:
                f.write(f"{eid}\t{tid}\tname_blocking\n")
                total_written += 1
    print(f"  wrote source2 candidates ({total_written:,} rows so far)")

    # This del is the actual fix for the MemoryError: union_s2 (millions of
    # candidate-id strings) and the full s2 DataFrame were staying alive for
    # the rest of the run while source3's exact/token dicts of similar size
    # got built on top of them. Free them before source3 starts.
    del s2, union_s2
    gc.collect()

    # ---- source3: same pattern ----
    s3["name_norm"] = s3["business_name"].apply(core_name)
    union_s3 = run_pair(s1_norm, s3, "name_norm", gt_s3, "source3", len(s1))
    diagnose_zero_candidates(s1, s3, union_s3, gt_s3, "source3")
    with open(out_path, "a") as f:
        for eid, cands in union_s3.items():
            for tid in cands:
                f.write(f"{eid}\t{tid}\tname_blocking\n")
                total_written += 1
    del s3, union_s3
    gc.collect()

    print(f"\nWrote candidate_pairs.tsv ({total_written:,} candidate pairs)")


if __name__ == "__main__":
    main()