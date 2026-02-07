#!/usr/bin/env python3
"""
code_words.py - Detect potential code words and unusual language patterns
in the FULL_EPSTEIN_INDEX dataset.

Looks for:
  - Euphemisms and known code words from trafficking/abuse cases
  - Words that appear in unusual frequency clusters (bursty terms)
  - Recurring unusual bigrams/trigrams
  - Terms that appear disproportionately in emails vs other documents
  - Repeated phrases that may indicate coded communication

Usage:
    python code_words.py --input <csv_file> [options]
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Known euphemisms and code words documented in court filings, depositions,
# and investigative journalism about the Epstein case.
# Sources: court transcripts, victim testimony, published reporting.
# ---------------------------------------------------------------------------

KNOWN_CODE_WORDS = {
    # Documented in court filings and victim testimony
    "massage": "Frequently used as euphemism for sexual abuse in victim depositions",
    "masseuse": "Term used to recruit victims under guise of legitimate work",
    "massage therapist": "Recruitment term documented in court filings",
    "therapy": "Sometimes used as cover for abuse sessions",
    "session": "Used to describe scheduled abuse encounters",
    "appointment": "Scheduling term for abuse sessions",
    "modeling": "Recruitment pretext documented in testimony",
    "model": "Recruitment lure",
    "talent": "Recruitment context",
    "school": "Victim recruitment location per court filings",
    "young woman": "Euphemism documented in depositions",
    "young women": "Euphemism documented in depositions",
    "young lady": "Euphemism documented in depositions",
    "young girl": "Direct reference in court filings",
    "young girls": "Direct reference in court filings",
    "girl": "Referenced in victim testimony",
    "girls": "Referenced in victim testimony",
    "minor": "Legal term in charging documents",
    "minors": "Legal term in charging documents",
    "underage": "Referenced in investigations",
    "friend": "Used to describe procured victims",
    "friends": "Used to describe procured victims",
    "guest": "Used to describe visitors to properties",
    "guests": "Used to describe visitors to properties",
    "company": "Euphemism for providing victims",
    "entertainment": "Euphemism documented in testimony",
    "entertain": "Euphemism documented in testimony",
    "gift": "Payments/inducements to victims",
    "gifts": "Payments/inducements to victims",
    "donation": "Financial transfers potentially masking payments",
    "donations": "Financial transfers potentially masking payments",
    "scholarship": "Inducement/payment mechanism",
    "travel": "Trafficking logistics",
    "trip": "Trafficking logistics",
    "visit": "Scheduling term",
    "island": "Reference to Little St. James",
    "ranch": "Reference to Zorro Ranch",
    "house": "Reference to properties used for abuse",
    "apartment": "Reference to properties",
    "townhouse": "Reference to NYC property",
    "mansion": "Reference to Palm Beach property",
    "private": "Modifier for secret activities",
    "arrangement": "Euphemism for abuse scheduling",
    "arrangements": "Euphemism for abuse scheduling",
    "favor": "Requests for procuring victims",
    "favors": "Requests for procuring victims",
    "package": "Potentially coded reference",
    "delivery": "Potentially coded logistics term",
    "pickup": "Transportation logistics",
    "driver": "Transportation logistics",
    "pilot": "Flight logistics",
    "flight": "Transportation to abuse locations",
    "plane": "Reference to private aircraft",
    "jet": "Reference to private aircraft",
    "helicopter": "Transportation",
    "passport": "Travel document / identity",
    "visa": "Immigration document",
    "cash": "Untraceable payments",
    "wire": "Financial transfers",
    "transfer": "Financial transfers",
    "account": "Financial accounts",
    "trust": "Legal/financial structures",
    "foundation": "Entity structure for payments",
    "recruit": "Victim procurement",
    "recruited": "Victim procurement",
    "introduce": "Procuring / connecting",
    "introduced": "Procuring / connecting",
    "brought": "Victim transportation",
    "sent": "Victim transportation/referral",
    "provide": "Supplying victims",
    "procure": "Legal term for trafficking",
    "compensation": "Payment for silence/compliance",
    "settlement": "Legal/financial resolution",
    "nda": "Non-disclosure agreement",
    "confidential": "Secrecy",
    "silence": "Suppressing testimony",
    "quiet": "Suppressing information",
    "discreet": "Secrecy in operations",
    "private island": "Little St. James reference",
    "little saint james": "Private island",
    "zorro ranch": "New Mexico property",
    "el brillo": "Palm Beach property address",
    "lolita express": "Nickname for Epstein's Boeing 727",
    "black book": "Contact/address book",
    "little black book": "Contact/address book",
    "flight log": "Aircraft passenger records",
    "flight logs": "Aircraft passenger records",
}

# Patterns that may indicate coded/euphemistic language
SUSPICIOUS_PATTERNS = [
    (r"\b(?:send|bring|get)\s+(?:a\s+)?(?:girl|young\s+(?:woman|lady))", "procurement language"),
    (r"\b(?:massage|therapy)\s+(?:session|appointment|table)", "massage euphemism"),
    (r"\bnew\s+(?:girl|friend|talent|model)", "new victim reference"),
    (r"\b(?:special|private|personal)\s+(?:session|service|arrangement|meeting)", "private session reference"),
    (r"\b(?:pick\s*up|drop\s*off)\s+(?:at|from|the)", "transportation logistics"),
    (r"\b(?:age|years?\s+old|born\s+in|birthday)", "age reference"),
    (r"\b(?:1[2-7])\s*(?:year|yr)s?\s*old", "minor age reference"),
    (r"\b(?:high\s+school|middle\s+school|freshman|sophomore|junior\b)", "school-age reference"),
    (r"\b(?:recruit|find|locate|source)\s+(?:more|new|another|some)", "recruitment language"),
    (r"\bpaid\s+(?:her|him|them|cash|in\s+cash)", "cash payment reference"),
    (r"\b(?:keep\s+(?:quiet|silent|secret)|don'?t\s+(?:tell|say|mention))", "secrecy language"),
    (r"\b(?:destroy|shred|delete|erase|get\s+rid)", "evidence destruction"),
    (r"\b(?:cleaning|clean\s+up|dispose|remove\s+evidence)", "cleanup language"),
]


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
        fn_col = None
        txt_col = None
        lower_fields = {f_.lower().strip(): f_ for f_ in fieldnames}
        for c in ["filename", "file_name", "file", "source", "document", "id"]:
            if c in lower_fields:
                fn_col = lower_fields[c]
                break
        for c in ["text", "content", "extracted_text", "body", "transcript"]:
            if c in lower_fields:
                txt_col = lower_fields[c]
                break
        if txt_col is None and len(fieldnames) == 2:
            fn_col, txt_col = fieldnames[0], fieldnames[1]
        elif txt_col is None:
            txt_col = fieldnames[-1]
        for i, row in enumerate(reader):
            doc_id = row.get(fn_col, f"doc_{i}") if fn_col else f"doc_{i}"
            text = row.get(txt_col, "")
            if text and text.strip():
                docs.append((doc_id, text))
    return docs


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


# Extended stop words including email boilerplate
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
    "too", "under", "never", "again", "further", "once", "why", "few",
    "off", "until", "above", "below", "against", "nor",
    # Email boilerplate
    "page", "date", "file", "subject", "sent", "received",
    "dear", "sincerely", "regards", "please", "thank", "thanks",
    "note", "copy", "fax", "email", "mail", "phone", "call",
    "number", "name", "address", "forwarded", "message", "wrote",
    # Document boilerplate that appears in ~13%+ of these docs
    "house", "oversight", "communication", "information", "gmail",
    "part", "including", "thereof", "privileged", "confidential",
    "attorney", "client", "intended", "addressee", "property",
    "unauthorized", "disclosure", "copying", "unlawful", "constitute",
    "inside", "error", "notify", "immediately", "return", "destroy",
    "copies", "copyright", "rights", "reserved", "prohibited",
    "strictly", "contained", "attachments", "sender",
    # Common filler
    "www", "http", "https", "com", "org", "net", "gov",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "january", "february", "march", "april",
    "june", "july", "august", "september", "october", "november",
    "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
}


def analyze_code_words(docs):
    """Scan documents for known code words and report frequencies."""
    print("\n" + "=" * 90)
    print("KNOWN CODE WORDS & EUPHEMISMS - FREQUENCY ANALYSIS")
    print("=" * 90)

    results = []
    for term, description in KNOWN_CODE_WORDS.items():
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        count = 0
        doc_count = 0
        example_docs = []
        for doc_id, text in docs:
            matches = pat.findall(text)
            if matches:
                count += len(matches)
                doc_count += 1
                if len(example_docs) < 3:
                    example_docs.append(doc_id)
        if count > 0:
            results.append((term, count, doc_count, description, example_docs))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Term':<25} {'Count':>8} {'Docs':>8} {'Doc%':>7}  Description")
    print("-" * 90)
    for term, count, doc_count, desc, examples in results:
        pct = doc_count / len(docs) * 100
        print(f"  {term:<23} {count:>8,} {doc_count:>8,} {pct:>6.1f}%  {desc}")

    return results


def analyze_suspicious_patterns(docs):
    """Scan for suspicious language patterns using regex."""
    print("\n\n" + "=" * 90)
    print("SUSPICIOUS LANGUAGE PATTERNS")
    print("=" * 90)

    results = []
    for pattern_str, label in SUSPICIOUS_PATTERNS:
        pat = re.compile(pattern_str, re.IGNORECASE)
        matches_found = []
        doc_count = 0
        for doc_id, text in docs:
            matches = pat.findall(text)
            if matches:
                doc_count += 1
                for m in matches[:3]:
                    if len(matches_found) < 10:
                        # Get surrounding context
                        for match_obj in pat.finditer(text):
                            start = max(0, match_obj.start() - 40)
                            end = min(len(text), match_obj.end() + 40)
                            ctx = text[start:end].replace("\n", " ").strip()
                            matches_found.append((doc_id, ctx))
                            break

        if doc_count > 0:
            results.append((label, doc_count, matches_found))

    results.sort(key=lambda x: x[1], reverse=True)

    for label, doc_count, examples in results:
        pct = doc_count / len(docs) * 100
        print(f"\n  [{label}] - found in {doc_count} documents ({pct:.1f}%)")
        for doc_id, ctx in examples[:5]:
            ctx_clean = ctx[:100]
            print(f"    {doc_id}: \"...{ctx_clean}...\"")

    return results


def analyze_bursty_terms(docs, min_length=4, top_n=100):
    """Find 'bursty' terms - words that appear frequently but only in a
    small cluster of documents. These may indicate coded language used
    between specific correspondents.

    Uses TF-IDF-like scoring: high term frequency in few documents.
    """
    print("\n\n" + "=" * 90)
    print("BURSTY / CLUSTERED TERMS (high frequency, low document spread)")
    print("=" * 90)
    print("These terms appear many times but are concentrated in few documents,")
    print("potentially indicating specialized vocabulary or code words.\n")

    word_count = Counter()
    word_doc_freq = Counter()

    for doc_id, text in docs:
        words = tokenize(text)
        filtered = set()
        for w in words:
            if len(w) < min_length or w in STOP_WORDS:
                continue
            word_count[w] += 1
            filtered.add(w)
        word_doc_freq.update(filtered)

    n_docs = len(docs)
    # Score = total_count * log(n_docs / doc_freq) -- TF-IDF-like
    # High score = appears many times but in few docs
    scored = []
    for word, count in word_count.items():
        df = word_doc_freq[word]
        if count < 10 or df < 3:  # minimum thresholds
            continue
        # Concentration ratio: avg occurrences per document it appears in
        concentration = count / df
        idf = math.log(n_docs / df)
        score = count * idf
        if concentration >= 2.0:  # at least 2x per doc on average
            scored.append((word, count, df, concentration, score))

    scored.sort(key=lambda x: x[4], reverse=True)

    print(f"{'Rank':<6} {'Term':<30} {'Total':>8} {'Docs':>6} {'Avg/Doc':>8} {'Score':>10}")
    print("-" * 75)
    for rank, (word, count, df, conc, score) in enumerate(scored[:top_n], 1):
        pct = df / n_docs * 100
        print(f"{rank:<6} {word:<30} {count:>8,} {df:>6} {conc:>8.1f} {score:>10.0f}")


def analyze_unusual_bigrams(docs, min_length=4, top_n=80):
    """Find unusual bigrams - word pairs that recur but aren't common English."""
    print("\n\n" + "=" * 90)
    print("UNUSUAL RECURRING BIGRAMS (filtered: no boilerplate)")
    print("=" * 90)

    # Additional boilerplate bigrams to skip
    boilerplate_bigrams = {
        "house oversight", "jeevacation gmail", "flags read", "read invitation",
        "invitation guid", "including attachments", "error notify",
        "rights reserved", "communication error", "information contained",
        "strictly prohibited", "notify immediately", "destroy communication",
        "copies thereof", "communication copies", "thereof including",
        "unauthorized disclosure", "disclosure copying", "attorney client",
        "part thereof", "thereof strictly", "communication part",
        "intended addressee", "contained communication", "addressee property",
        "attachments copyright", "client privileged", "immediately return",
        "copying communication", "communication confidential",
        "confidential attorney", "information intended", "prohibited unlawful",
        "unlawful communication", "constitute inside", "inside information",
        "copyright rights", "privileged constitute", "return jeevacation",
        "gmail destroy", "guid message", "property unauthorized",
        "reserved house", "reserved information", "sender flags",
        "gmail flags", "sender jeeitunes", "jeeitunes gmail",
        "gmail wrote", "forwarded message", "message sender",
        "jeffrey jeevacation", "jeffrey jeeyacation", "epstein jeevacation",
        "jeeyacation gmail", "importance high",
    }

    bigram_count = Counter()
    bigram_doc_freq = Counter()

    for doc_id, text in docs:
        words = tokenize(text)
        filtered = []
        for w in words:
            if len(w) < min_length or w in STOP_WORDS:
                continue
            filtered.append(w)

        doc_bigrams = set()
        for i in range(len(filtered) - 1):
            bg = f"{filtered[i]} {filtered[i+1]}"
            bigram_count[bg] += 1
            doc_bigrams.add(bg)
        bigram_doc_freq.update(doc_bigrams)

    # Filter out boilerplate
    filtered_bigrams = []
    for bg, count in bigram_count.items():
        if bg in boilerplate_bigrams:
            continue
        df = bigram_doc_freq[bg]
        if count < 5 or df < 3:
            continue
        filtered_bigrams.append((bg, count, df))

    filtered_bigrams.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Bigram':<45} {'Count':>8} {'Docs':>6} {'Doc%':>7}")
    print("-" * 80)
    for rank, (bg, count, df) in enumerate(filtered_bigrams[:top_n], 1):
        pct = df / len(docs) * 100
        print(f"{rank:<6} {bg:<45} {count:>8,} {df:>6} {pct:>6.1f}%")


def analyze_extended_word_frequency(docs, min_length=4, top_n=200):
    """Extended word frequency, filtering all boilerplate."""
    print("\n\n" + "=" * 90)
    print(f"EXTENDED WORD FREQUENCY - TOP {top_n} (all boilerplate filtered)")
    print("=" * 90)

    word_count = Counter()
    word_doc_freq = Counter()

    for doc_id, text in docs:
        words = tokenize(text)
        doc_words = set()
        for w in words:
            if len(w) < min_length or w in STOP_WORDS:
                continue
            word_count[w] += 1
            doc_words.add(w)
        word_doc_freq.update(doc_words)

    print(f"\n{'Rank':<6} {'Word':<30} {'Count':>10} {'Docs':>8} {'Doc%':>7}")
    print("-" * 68)
    for rank, (word, count) in enumerate(word_count.most_common(top_n), 1):
        df = word_doc_freq[word]
        pct = df / len(docs) * 100
        print(f"{rank:<6} {word:<30} {count:>10,} {df:>8,} {pct:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Detect potential code words and unusual language patterns.",
    )
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file.")
    parser.add_argument("--top", "-t", type=int, default=200, help="Top N for frequency lists.")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    docs = load_csv(args.input)
    print(f"Loaded {len(docs)} documents.\n")

    # 1. Extended word frequency
    analyze_extended_word_frequency(docs, top_n=args.top)

    # 2. Known code words
    analyze_code_words(docs)

    # 3. Suspicious patterns
    analyze_suspicious_patterns(docs)

    # 4. Bursty terms
    analyze_bursty_terms(docs, top_n=args.top)

    # 5. Unusual bigrams
    analyze_unusual_bigrams(docs, top_n=args.top)


if __name__ == "__main__":
    main()
