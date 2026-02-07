#!/usr/bin/env python3
"""
find_connections.py - Entity connection finder for the FULL_EPSTEIN_INDEX dataset.

Analyzes extracted text from declassified Epstein documents to identify
co-occurrences of known persons, organizations, and locations across documents.
Outputs a connection graph showing which entities appear together and how often.

Usage:
    python find_connections.py --input <csv_file> [options]

    If no --input is provided, tries to load from HuggingFace:
        theelderemo/FULL_EPSTEIN_INDEX

Examples:
    python find_connections.py --input data.csv
    python find_connections.py --input data.csv --min-weight 3 --format json
    python find_connections.py --input data.csv --entity "Ghislaine Maxwell"
    python find_connections.py --input data.csv --top 50
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# ---------------------------------------------------------------------------
# Known entities compiled from public reporting on the Epstein case.
# These names come from court filings, flight logs, and government releases
# that are already in the public domain.
# ---------------------------------------------------------------------------

KNOWN_PERSONS = [
    "Jeffrey Epstein",
    "Ghislaine Maxwell",
    "Jean-Luc Brunel",
    "Sarah Kellen",
    "Nadia Marcinkova",
    "Lesley Groff",
    "Adriana Ross",
    "Les Wexner",
    "Leslie Wexner",
    "Alan Dershowitz",
    "Prince Andrew",
    "Duke of York",
    "Bill Clinton",
    "William Clinton",
    "Donald Trump",
    "Bill Richardson",
    "George Mitchell",
    "Ehud Barak",
    "Glenn Dubin",
    "Eva Dubin",
    "Marvin Minsky",
    "Stephen Hawking",
    "Larry Summers",
    "Leon Black",
    "Jes Staley",
    "Bill Gates",
    "William Gates",
    "Reid Hoffman",
    "Mort Zuckerman",
    "Woody Allen",
    "David Copperfield",
    "Naomi Campbell",
    "Kevin Spacey",
    "Chris Tucker",
    "Courtney Love",
    "Virginia Giuffre",
    "Virginia Roberts",
    "Carolyn Andriano",
    "Annie Farmer",
    "Maria Farmer",
    "Haley Robson",
    "Johanna Sjoberg",
    "Sarah Ransome",
    "Chauntae Davies",
    "Alexander Acosta",
    "Barry Krischer",
    "Robert Mueller",
    "James Comey",
    "Michael Reiter",
    "Joseph Recarey",
    "Larry Visoski",
    "David Rodgers",
    "Juan Alessi",
    "Alfredo Rodriguez",
    "Emmy Tayler",
    "Robert Maxwell",
    "Isabel Maxwell",
    "Kevin Maxwell",
    "Christine Maxwell",
    "Darren Indyke",
    "Richard Kahn",
    "Mark Epstein",
    "Gerald Lefcourt",
    "Jay Lefkowitz",
    "Kenneth Starr",
    "Roy Black",
    "Jack Goldberger",
    "Guy Lewis",
    "Martin Weinberg",
    "Laura Goldman",
    "Peter Listerman",
    "Cindy Lopez",
    "Tony Figueroa",
    "Igor Zinoviev",
    "Steven Hoffenberg",
    "Celina Midelfart",
    "Eva Andersson-Dubin",
    "Glenn Dubin",
    "Lynn Forester de Rothschild",
    "Evelyn de Rothschild",
    "Tom Barrack",
    "Leon Botstein",
    "Henry Rosovsky",
    "Stuart Pivar",
    "Peggy Siegal",
    "Peter Mandelson",
    "Terri Shields",
    "Emmy Tayler",
    "Shelley Lewis",
    "Doug Band",
    "David Boies",
    "Sigrid McCawley",
    "Brad Edwards",
    "Paul Cassell",
    "Jeffrey Pagliuca",
    "Laura Menninger",
    "Christian Everdell",
    "Alison Moe",
    "Maurene Comey",
    "Alex Rossmiller",
    "Audrey Strauss",
    "James Patterson",
    "Julie Brown",
    "Conchita Sarnoff",
    "Michael Cernovich",
    "Mike Fisten",
]

KNOWN_ORGANIZATIONS = [
    "J. Epstein VI Foundation",
    "Epstein VI Foundation",
    "Southern Trust",
    "Southern Trust Company",
    "Financial Trust Company",
    "Butterfly Trust",
    "C.O.U.Q. Foundation",
    "COUQ Foundation",
    "Gratitude America",
    "Program for Evolutionary Dynamics",
    "Harvard University",
    "MIT Media Lab",
    "Massachusetts Institute of Technology",
    "Dalton School",
    "Bear Stearns",
    "Deutsche Bank",
    "JPMorgan",
    "JP Morgan",
    "JPMorgan Chase",
    "Towers Financial",
    "L Brands",
    "Victoria's Secret",
    "The Limited",
    "Wexner Foundation",
    "Ohio State University",
    "Palm Beach Police",
    "FBI",
    "Federal Bureau of Investigation",
    "DOJ",
    "Department of Justice",
    "Southern District of New York",
    "SDNY",
    "Bureau of Prisons",
    "BOP",
    "Metropolitan Correctional Center",
    "MCC",
    "U.S. Virgin Islands",
    "USVI",
    "Office of the Attorney General",
    "House Committee on Oversight",
    "Customs and Border",
    "Secret Service",
    "Scotland Yard",
    "Interpol",
    "Maxwell Foundation",
    "TerraMar Project",
    "Clinton Foundation",
    "Clinton Global Initiative",
    "Lolita Express",
    "Apollo Global Management",
    "Barclays",
]

KNOWN_LOCATIONS = [
    "Little St. James",
    "Little Saint James",
    "Great St. James",
    "Great Saint James",
    "Zorro Ranch",
    "Stanley, New Mexico",
    "El Brillo Way",
    "Palm Beach",
    "East 71st Street",
    "New York mansion",
    "Paris apartment",
    "Avenue Foch",
    "Teterboro",
    "Columbus, Ohio",
    "Virgin Islands",
    "Caribbean",
    "Mar-a-Lago",
]


def build_entity_patterns(entities):
    """Compile regex patterns for a list of entity names.

    Returns list of (compiled_pattern, canonical_name) tuples.
    Uses word boundaries to avoid partial matches.
    """
    patterns = []
    seen = set()
    for name in entities:
        canonical = name.strip()
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        # Escape special regex characters, use word boundaries
        escaped = re.escape(canonical)
        pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
        patterns.append((pat, canonical))
    return patterns


def extract_entities(text, patterns):
    """Return set of canonical entity names found in text."""
    found = set()
    for pat, canonical in patterns:
        if pat.search(text):
            found.add(canonical)
    return found


def detect_csv_columns(reader):
    """Auto-detect which columns contain filename and text content.

    Returns (filename_col, text_col) as column name strings.
    """
    fieldnames = reader.fieldnames
    if not fieldnames:
        return None, None

    fn_col = None
    txt_col = None
    lower_fields = {f.lower().strip(): f for f in fieldnames}

    # Try common filename column names
    for candidate in ["filename", "file_name", "file", "source", "document", "doc_id", "id"]:
        if candidate in lower_fields:
            fn_col = lower_fields[candidate]
            break

    # Try common text column names
    for candidate in ["text", "content", "extracted_text", "body", "transcript", "ocr_text"]:
        if candidate in lower_fields:
            txt_col = lower_fields[candidate]
            break

    # Fallback: if only two columns, assume first is filename, second is text
    if fn_col is None and txt_col is None and len(fieldnames) == 2:
        fn_col = fieldnames[0]
        txt_col = fieldnames[1]

    # If still no text column, pick the one with the longest average values
    if txt_col is None:
        txt_col = fieldnames[-1]  # last column as fallback

    return fn_col, txt_col


def load_csv(path):
    """Load CSV and return list of (doc_id, text) tuples."""
    docs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # Sniff delimiter
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = "excel"

        reader = csv.DictReader(f, dialect=dialect)
        fn_col, txt_col = detect_csv_columns(reader)

        if txt_col is None:
            print(f"Error: Could not detect text column. Columns found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        print(f"Detected columns - filename: {fn_col or '(none)'}, text: {txt_col}")
        print(f"All columns: {reader.fieldnames}")

        for i, row in enumerate(reader):
            doc_id = row.get(fn_col, f"doc_{i}") if fn_col else f"doc_{i}"
            text = row.get(txt_col, "")
            if text and text.strip():
                docs.append((doc_id, text))

    return docs


def load_huggingface():
    """Try loading the dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        print("Loading dataset from HuggingFace: theelderemo/FULL_EPSTEIN_INDEX ...")
        ds = load_dataset("theelderemo/FULL_EPSTEIN_INDEX", split="train")
        docs = []
        columns = ds.column_names
        print(f"Dataset columns: {columns}")

        # Detect text column
        txt_col = None
        fn_col = None
        for c in columns:
            cl = c.lower()
            if cl in ("text", "content", "extracted_text", "body"):
                txt_col = c
            if cl in ("filename", "file_name", "file", "source"):
                fn_col = c

        if txt_col is None:
            txt_col = columns[-1]
        if fn_col is None and len(columns) >= 2:
            fn_col = columns[0]

        print(f"Using columns - filename: {fn_col or '(auto)'}, text: {txt_col}")

        for i, row in enumerate(ds):
            doc_id = row.get(fn_col, f"doc_{i}") if fn_col else f"doc_{i}"
            text = row.get(txt_col, "")
            if text and str(text).strip():
                docs.append((doc_id, str(text)))

        return docs
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}", file=sys.stderr)
        return None


def find_connections(docs, person_patterns, org_patterns, loc_patterns, min_weight=1):
    """Analyze documents to find entity co-occurrence connections.

    Returns:
        edges: dict of (entity_a, entity_b) -> {weight, documents}
        entity_docs: dict of entity -> set of doc_ids
        entity_types: dict of entity -> type string
    """
    all_patterns = (
        [(p, n, "person") for p, n in person_patterns]
        + [(p, n, "organization") for p, n in org_patterns]
        + [(p, n, "location") for p, n in loc_patterns]
    )

    entity_docs = defaultdict(set)  # entity -> set of doc_ids
    entity_types = {}  # entity -> type
    doc_entities = {}  # doc_id -> set of entities

    total = len(docs)
    for idx, (doc_id, text) in enumerate(docs):
        if idx % 500 == 0:
            print(f"  Processing document {idx + 1}/{total} ...", end="\r")

        found = set()
        for pat, canonical, etype in all_patterns:
            if pat.search(text):
                found.add(canonical)
                entity_types[canonical] = etype
                entity_docs[canonical].add(doc_id)

        doc_entities[doc_id] = found

    print(f"  Processed {total} documents.                    ")

    # Build co-occurrence edges
    edges = defaultdict(lambda: {"weight": 0, "documents": []})
    for doc_id, entities in doc_entities.items():
        if len(entities) < 2:
            continue
        for a, b in combinations(sorted(entities), 2):
            key = (a, b)
            edges[key]["weight"] += 1
            if len(edges[key]["documents"]) < 10:  # keep up to 10 example docs
                edges[key]["documents"].append(doc_id)

    # Filter by minimum weight
    if min_weight > 1:
        edges = {k: v for k, v in edges.items() if v["weight"] >= min_weight}

    return edges, entity_docs, entity_types


def filter_by_entity(edges, entity_docs, target_entity):
    """Filter connections to only those involving a specific entity."""
    target_lower = target_entity.lower()

    # Find canonical name
    canonical = None
    for name in entity_docs:
        if name.lower() == target_lower or target_lower in name.lower():
            canonical = name
            break

    if canonical is None:
        print(f"Entity '{target_entity}' not found in any documents.", file=sys.stderr)
        return {}

    filtered = {}
    for (a, b), data in edges.items():
        if a == canonical or b == canonical:
            filtered[(a, b)] = data

    return filtered


def output_text(edges, entity_docs, entity_types, top_n=None):
    """Print connections as formatted text."""
    sorted_edges = sorted(edges.items(), key=lambda x: x[1]["weight"], reverse=True)
    if top_n:
        sorted_edges = sorted_edges[:top_n]

    print("\n" + "=" * 80)
    print("ENTITY CONNECTIONS")
    print("=" * 80)

    # Summary stats
    all_entities = set()
    for (a, b) in edges:
        all_entities.add(a)
        all_entities.add(b)

    persons = [e for e in all_entities if entity_types.get(e) == "person"]
    orgs = [e for e in all_entities if entity_types.get(e) == "organization"]
    locs = [e for e in all_entities if entity_types.get(e) == "location"]

    print(f"\nEntities found: {len(all_entities)} total")
    print(f"  Persons:       {len(persons)}")
    print(f"  Organizations: {len(orgs)}")
    print(f"  Locations:     {len(locs)}")
    print(f"Connections:     {len(edges)}")
    print()

    # Top mentioned entities
    entity_counts = Counter()
    for entity, doc_set in entity_docs.items():
        entity_counts[entity] = len(doc_set)

    print("TOP MENTIONED ENTITIES:")
    print("-" * 50)
    for entity, count in entity_counts.most_common(30):
        etype = entity_types.get(entity, "?")
        print(f"  {entity:<40} [{etype:<12}] in {count} documents")

    print("\n\nCONNECTIONS (by co-occurrence strength):")
    print("-" * 80)

    for (a, b), data in sorted_edges:
        weight = data["weight"]
        bar = "#" * min(weight, 50)
        type_a = entity_types.get(a, "?")
        type_b = entity_types.get(b, "?")
        print(f"\n  {a} [{type_a}] <-> {b} [{type_b}]")
        print(f"    Co-occurrences: {weight}  {bar}")
        if data["documents"]:
            print(f"    Example docs:   {', '.join(data['documents'][:5])}")


def output_json(edges, entity_docs, entity_types, top_n=None):
    """Output connections as JSON."""
    sorted_edges = sorted(edges.items(), key=lambda x: x[1]["weight"], reverse=True)
    if top_n:
        sorted_edges = sorted_edges[:top_n]

    nodes = {}
    for entity, doc_set in entity_docs.items():
        nodes[entity] = {
            "type": entity_types.get(entity, "unknown"),
            "document_count": len(doc_set),
        }

    connections = []
    for (a, b), data in sorted_edges:
        connections.append({
            "source": a,
            "target": b,
            "weight": data["weight"],
            "example_documents": data["documents"][:5],
        })

    result = {
        "summary": {
            "total_entities": len(nodes),
            "total_connections": len(connections),
        },
        "entities": nodes,
        "connections": connections,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


def output_csv_format(edges, entity_types, top_n=None):
    """Output connections as CSV to stdout."""
    sorted_edges = sorted(edges.items(), key=lambda x: x[1]["weight"], reverse=True)
    if top_n:
        sorted_edges = sorted_edges[:top_n]

    writer = csv.writer(sys.stdout)
    writer.writerow(["source", "source_type", "target", "target_type", "weight", "example_documents"])

    for (a, b), data in sorted_edges:
        writer.writerow([
            a,
            entity_types.get(a, "unknown"),
            b,
            entity_types.get(b, "unknown"),
            data["weight"],
            "; ".join(data["documents"][:5]),
        ])


def output_gexf(edges, entity_docs, entity_types, filepath="connections.gexf"):
    """Export as GEXF graph file (for Gephi visualization)."""
    if not HAS_NETWORKX:
        print("networkx is required for GEXF export. Install with: pip install networkx", file=sys.stderr)
        return

    G = nx.Graph()

    for entity, doc_set in entity_docs.items():
        G.add_node(entity, type=entity_types.get(entity, "unknown"), document_count=len(doc_set))

    for (a, b), data in edges.items():
        G.add_edge(a, b, weight=data["weight"])

    nx.write_gexf(G, filepath)
    print(f"Graph exported to {filepath} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def main():
    parser = argparse.ArgumentParser(
        description="Find entity connections in Epstein document corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to CSV file with extracted text. If omitted, tries HuggingFace.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "csv", "gexf"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--min-weight", "-w",
        type=int,
        default=1,
        help="Minimum co-occurrence count to include a connection (default: 1).",
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=None,
        help="Show only the top N connections by weight.",
    )
    parser.add_argument(
        "--entity", "-e",
        type=str,
        default=None,
        help="Filter connections to only those involving this entity.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (for gexf format). Default: connections.gexf.",
    )
    parser.add_argument(
        "--extra-names",
        type=str,
        default=None,
        help="Path to a text file with additional person names (one per line).",
    )

    args = parser.parse_args()

    # Load documents
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        docs = load_csv(args.input)
    else:
        docs = load_huggingface()
        if docs is None:
            print(
                "No input provided and HuggingFace loading failed.\n"
                "Please provide a CSV file with --input <path>.\n\n"
                "You can download the dataset from:\n"
                "  https://huggingface.co/datasets/theelderemo/FULL_EPSTEIN_INDEX\n",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loaded {len(docs)} documents.\n")

    # Build entity lists
    persons = list(KNOWN_PERSONS)
    if args.extra_names:
        with open(args.extra_names, "r") as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    persons.append(name)
        print(f"Loaded {len(persons)} person names (including extras).")

    # Compile patterns
    print("Compiling entity patterns ...")
    person_patterns = build_entity_patterns(persons)
    org_patterns = build_entity_patterns(KNOWN_ORGANIZATIONS)
    loc_patterns = build_entity_patterns(KNOWN_LOCATIONS)

    total_patterns = len(person_patterns) + len(org_patterns) + len(loc_patterns)
    print(f"  {len(person_patterns)} person patterns")
    print(f"  {len(org_patterns)} organization patterns")
    print(f"  {len(loc_patterns)} location patterns")
    print(f"  {total_patterns} total\n")

    # Find connections
    print("Scanning documents for entity co-occurrences ...")
    edges, entity_docs, entity_types = find_connections(
        docs, person_patterns, org_patterns, loc_patterns, min_weight=args.min_weight
    )

    if not edges:
        print("\nNo connections found with the current settings.")
        print("Try lowering --min-weight or checking your input data.")
        sys.exit(0)

    # Filter by entity if requested
    if args.entity:
        edges = filter_by_entity(edges, entity_docs, args.entity)
        if not edges:
            sys.exit(0)

    # Output
    if args.format == "text":
        output_text(edges, entity_docs, entity_types, top_n=args.top)
    elif args.format == "json":
        output_json(edges, entity_docs, entity_types, top_n=args.top)
    elif args.format == "csv":
        output_csv_format(edges, entity_types, top_n=args.top)
    elif args.format == "gexf":
        outpath = args.output or "connections.gexf"
        output_gexf(edges, entity_docs, entity_types, filepath=outpath)


if __name__ == "__main__":
    main()
