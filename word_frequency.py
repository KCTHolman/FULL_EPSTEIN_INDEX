#!/usr/bin/env python3
"""
word_frequency.py - Word frequency analysis for the FULL_EPSTEIN_INDEX dataset.

Analyzes extracted text to find the most frequently used words, with options
to filter out common English stop words and short words to surface meaningful terms.

Usage:
    python word_frequency.py --input <csv_file> [options]

Examples:
    python word_frequency.py --input data.csv
    python word_frequency.py --input data.csv --top 100 --min-length 4
    python word_frequency.py --input data.csv --bigrams --top 50
    python word_frequency.py --input data.csv --format json
"""

import argparse
import csv
import json
import re
import sys
import os
from collections import Counter

# Common English stop words to filter out
STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "are", "was", "were", "been", "has",
    "had", "did", "does", "is", "am", "being", "having", "doing",
    "will", "shall", "should", "would", "could", "might", "may",
    "must", "need", "dare", "ought", "used", "going", "able",
    "said", "more", "very", "through", "where", "much", "before",
    "between", "each", "while", "such", "here", "those", "own",
    "same", "down", "been", "still", "both", "during", "may",
    "did", "too", "under", "never", "should", "very", "after",
    "without", "again", "further", "once", "why", "both", "few",
    "off", "until", "above", "below", "against", "nor",
    # OCR noise / common document words
    "page", "date", "file", "subject", "sent", "received",
    "dear", "dear", "sincerely", "regards", "please", "thank",
    "thanks", "note", "copy", "fax", "email", "mail", "phone",
    "call", "number", "name", "address", "re", "cc", "bcc",
    "www", "http", "https", "com", "org", "net", "gov",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "january", "february", "march", "april",
    "june", "july", "august", "september", "october", "november",
    "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "am", "pm",
}


def load_csv(path):
    """Load CSV and return list of (doc_id, text) tuples."""
    docs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = "excel"

        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = reader.fieldnames

        # Auto-detect columns
        fn_col = None
        txt_col = None
        lower_fields = {f_.lower().strip(): f_ for f_ in fieldnames}

        for candidate in ["filename", "file_name", "file", "source", "document", "id"]:
            if candidate in lower_fields:
                fn_col = lower_fields[candidate]
                break

        for candidate in ["text", "content", "extracted_text", "body", "transcript"]:
            if candidate in lower_fields:
                txt_col = lower_fields[candidate]
                break

        if txt_col is None and len(fieldnames) == 2:
            fn_col = fieldnames[0]
            txt_col = fieldnames[1]
        elif txt_col is None:
            txt_col = fieldnames[-1]

        print(f"Columns: {fieldnames}")
        print(f"Using text column: {txt_col}")

        for i, row in enumerate(reader):
            doc_id = row.get(fn_col, f"doc_{i}") if fn_col else f"doc_{i}"
            text = row.get(txt_col, "")
            if text and text.strip():
                docs.append((doc_id, text))

    return docs


def tokenize(text):
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def count_words(docs, min_length=3, include_stop_words=False):
    """Count word frequencies across all documents."""
    counter = Counter()
    doc_freq = Counter()  # in how many documents does a word appear

    for doc_id, text in docs:
        words = tokenize(text)
        filtered = []
        for w in words:
            if len(w) < min_length:
                continue
            if not include_stop_words and w in STOP_WORDS:
                continue
            filtered.append(w)

        counter.update(filtered)
        doc_freq.update(set(filtered))

    return counter, doc_freq


def count_bigrams(docs, min_length=3, include_stop_words=False):
    """Count bigram (two-word pair) frequencies."""
    counter = Counter()
    doc_freq = Counter()

    for doc_id, text in docs:
        words = tokenize(text)
        filtered = []
        for w in words:
            if len(w) < min_length:
                continue
            if not include_stop_words and w in STOP_WORDS:
                continue
            filtered.append(w)

        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered) - 1)]
        counter.update(bigrams)
        doc_freq.update(set(bigrams))

    return counter, doc_freq


def count_trigrams(docs, min_length=3, include_stop_words=False):
    """Count trigram (three-word) frequencies."""
    counter = Counter()
    doc_freq = Counter()

    for doc_id, text in docs:
        words = tokenize(text)
        filtered = []
        for w in words:
            if len(w) < min_length:
                continue
            if not include_stop_words and w in STOP_WORDS:
                continue
            filtered.append(w)

        trigrams = [
            f"{filtered[i]} {filtered[i+1]} {filtered[i+2]}"
            for i in range(len(filtered) - 2)
        ]
        counter.update(trigrams)
        doc_freq.update(set(trigrams))

    return counter, doc_freq


def output_text(counter, doc_freq, total_docs, top_n, label="WORDS"):
    """Print frequency table as formatted text."""
    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} MOST FREQUENT {label}")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6} {'Term':<40} {'Count':>10} {'Docs':>8} {'Doc%':>8}")
    print("-" * 80)

    for rank, (word, count) in enumerate(counter.most_common(top_n), 1):
        df = doc_freq.get(word, 0)
        doc_pct = (df / total_docs * 100) if total_docs > 0 else 0
        bar = "#" * min(count // max(counter.most_common(top_n)[-1][1], 1), 30)
        print(f"{rank:<6} {word:<40} {count:>10,} {df:>8,} {doc_pct:>7.1f}%")


def output_json(counter, doc_freq, total_docs, top_n):
    """Output as JSON."""
    results = []
    for word, count in counter.most_common(top_n):
        df = doc_freq.get(word, 0)
        results.append({
            "term": word,
            "count": count,
            "document_frequency": df,
            "document_percentage": round(df / total_docs * 100, 2) if total_docs > 0 else 0,
        })

    output = {
        "total_documents": total_docs,
        "total_unique_terms": len(counter),
        "top_terms": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def output_csv_format(counter, doc_freq, total_docs, top_n):
    """Output as CSV."""
    writer = csv.writer(sys.stdout)
    writer.writerow(["rank", "term", "count", "document_frequency", "document_percentage"])
    for rank, (word, count) in enumerate(counter.most_common(top_n), 1):
        df = doc_freq.get(word, 0)
        doc_pct = round(df / total_docs * 100, 2) if total_docs > 0 else 0
        writer.writerow([rank, word, count, df, doc_pct])


def main():
    parser = argparse.ArgumentParser(
        description="Word frequency analysis for Epstein document corpus.",
    )
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file.")
    parser.add_argument("--top", "-t", type=int, default=50, help="Show top N terms (default: 50).")
    parser.add_argument("--min-length", "-l", type=int, default=3, help="Minimum word length (default: 3).")
    parser.add_argument("--include-stop-words", action="store_true", help="Include common English stop words.")
    parser.add_argument("--bigrams", action="store_true", help="Analyze two-word combinations.")
    parser.add_argument("--trigrams", action="store_true", help="Analyze three-word combinations.")
    parser.add_argument("--format", "-f", choices=["text", "json", "csv"], default="text", help="Output format.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    docs = load_csv(args.input)
    print(f"Loaded {len(docs)} documents.\n")

    if args.bigrams:
        print("Counting bigrams ...")
        counter, doc_freq = count_bigrams(docs, args.min_length, args.include_stop_words)
        label = "BIGRAMS (WORD PAIRS)"
    elif args.trigrams:
        print("Counting trigrams ...")
        counter, doc_freq = count_trigrams(docs, args.min_length, args.include_stop_words)
        label = "TRIGRAMS (3-WORD PHRASES)"
    else:
        print("Counting words ...")
        counter, doc_freq = count_words(docs, args.min_length, args.include_stop_words)
        label = "WORDS"

    print(f"Found {len(counter):,} unique terms.\n")

    if args.format == "text":
        output_text(counter, doc_freq, len(docs), args.top, label)
    elif args.format == "json":
        output_json(counter, doc_freq, len(docs), args.top)
    elif args.format == "csv":
        output_csv_format(counter, doc_freq, len(docs), args.top)


if __name__ == "__main__":
    main()
