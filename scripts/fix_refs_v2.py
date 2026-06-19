#!/usr/bin/env python3
import re

fp = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/FULL_MANUSCRIPT.md"
with open(fp, "r", encoding="utf-8") as f:
    text = f.read()

# Split body and references
m = re.search(r"(^References\s*\n)", text, re.MULTILINE)
if not m:
    m = re.search(r"(## References\s*\n)", text)
    if not m:
        print("ERROR: No References section found")
        exit(1)

body = text[:m.start()]
ref_header = m.group(1)
ref_text = text[m.end():]

# Parse all references: number -> full text
refs = {}
for num, entry in re.findall(r"^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)", ref_text, re.MULTILINE | re.DOTALL):
    refs[int(num)] = entry.strip()

print(f"Found {len(refs)} references")

# Build reference lookup: for each ref, store first author and year
ref_lookup = []
for num, entry in refs.items():
    first_line = entry.split("\n")[0]
    # Extract first author(s) before first comma or period
    author_part = first_line.split(",")[0] if "," in first_line else first_line.split(".")[0]
    # Extract year - try multiple patterns
    yr = ""
    # Pattern 1: (YYYY)
    ym = re.search(r"\((\d{4})[a-z]?\)", first_line)
    if ym:
        yr = ym.group(1)
    else:
        # Pattern 2: , YYYY. or , YYYY)
        ym2 = re.search(r",\s*(\d{4})[a-z]?\s*[\.\)]", first_line)
        if ym2:
            yr = ym2.group(1)
        else:
            # Pattern 3: and Lin, C.Y., 2023.
            ym3 = re.search(r",\s*(\d{4})\s*\.", first_line)
            if ym3:
                yr = ym3.group(1)
    ref_lookup.append((num, author_part.strip(), yr, entry))

def normalize(s):
    return s.lower().replace("-", "").replace(" ", "").replace(".", "")

def find_ref(author_query, year_query):
    """Find reference number by fuzzy matching author and exact year."""
    author_query = author_query.strip()
    candidates = []
    for num, author_str, yr, _ in ref_lookup:
        if yr == year_query:
            # Exact match on author string
            if author_query.lower() in author_str.lower() or author_str.lower() in author_query.lower():
                return num
            candidates.append((num, author_str))
    # Fuzzy match: check if normalized author is close
    aq_norm = normalize(author_query)
    for num, author_str in candidates:
        as_norm = normalize(author_str)
        # Check substring in either direction
        if aq_norm in as_norm or as_norm in aq_norm:
            return num
        # Check word-by-word for compound names
        aq_words = set(author_query.lower().split())
        as_words = set(author_str.lower().split())
        if aq_words & as_words:
            return num
    # Very fuzzy: allow 1-2 character differences in normalized strings
    for num, author_str in candidates:
        as_norm = normalize(author_str)
        # If lengths are similar and there's significant overlap
        if abs(len(aq_norm) - len(as_norm)) <= 2:
            # Check longest common substring-ish: if one is almost a substring of the other
            min_len = min(len(aq_norm), len(as_norm))
            matches = sum(1 for i in range(min_len) if i < len(aq_norm) and i < len(as_norm) and aq_norm[i] == as_norm[i])
            if matches >= min_len - 2:
                return num
    return None

# Find all citations in body
citations = []

# Pattern 1: [N] or [N,M] (already numbered)
for mo in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
    vals = [int(n.strip()) for n in mo.group(1).split(",")]
    citations.append((mo.start(), "num", vals, mo.group(0)))

# Pattern 2: [Author et al., YYYY; Author2 et al., YYYY]
for mo in re.finditer(r"\[([A-Za-z\-\s&]+?\s+et\s+al\.\s*,\s*\d{4}(?:;\s*[A-Za-z\-\s&]+?\s+et\s+al\.\s*,\s*\d{4})*)\]", body):
    inner = mo.group(1)
    ind = re.findall(r"([A-Za-z\-\s&]+?)\s+et\s+al\.\s*,\s*(\d{4})", inner)
    citations.append((mo.start(), "ay_bracket", [(a.strip(), y) for a, y in ind], mo.group(0)))

# Pattern 3: [Author & Author, YYYY] or [Author and Author, YYYY]
for mo in re.finditer(r"\[([A-Za-z\-\s]+?(?:\s+(?:&|and)\s+[A-Za-z\-\s]+?)?),\s*(\d{4})\]", body):
    citations.append((mo.start(), "simple_bracket", [(mo.group(1).strip(), mo.group(2))], mo.group(0)))

# Pattern 4: (Author et al., YYYY; Author2 et al., YYYY)
for mo in re.finditer(r"\(([A-Za-z\-\s&]+?\s+et\s+al\.\s*,\s*\d{4}(?:;\s*[A-Za-z\-\s&]+?\s+et\s+al\.\s*,\s*\d{4})*)\)", body):
    inner = mo.group(1)
    ind = re.findall(r"([A-Za-z\-\s&]+?)\s+et\s+al\.\s*,\s*(\d{4})", inner)
    citations.append((mo.start(), "ay_paren", [(a.strip(), y) for a, y in ind], mo.group(0)))

# Pattern 5: (Author & Author, YYYY) or (Author, YYYY)
for mo in re.finditer(r"\(([A-Za-z\-\s]+?(?:\s+(?:&|and)\s+[A-Za-z\-\s]+?)?),\s*(\d{4})\)", body):
    author_str, yr = mo.group(1).strip(), mo.group(2)
    rn = find_ref(author_str, yr)
    if rn:
        citations.append((mo.start(), "simple_paren", [(author_str, yr)], mo.group(0)))

# Sort by position
citations.sort(key=lambda x: x[0])
print(f"Found {len(citations)} citation groups")

# Build sequential mapping
old2new = {}
nxt = 1
unmatched = []

for pos, ctype, data, orig in citations:
    if ctype == "num":
        for old in data:
            if old not in old2new:
                old2new[old] = nxt
                nxt += 1
    else:
        for author_str, yr in data:
            rn = find_ref(author_str, yr)
            if rn and rn not in old2new:
                old2new[rn] = nxt
                nxt += 1
            elif rn is None:
                unmatched.append((author_str, yr, orig))

if unmatched:
    print(f"\nWARNING: {len(unmatched)} unmatched citations:")
    for a, y, o in unmatched:
        print(f"  - '{a}', {y} in: {o[:80]}...")

print(f"Mapped {len(old2new)} unique cited references")

# Replace citations in body (backwards to preserve positions)
new_body = body
for pos, ctype, data, orig in reversed(citations):
    if ctype == "num":
        nn = [str(old2new.get(d, d)) for d in data]
        rep = "[" + ",".join(nn) + "]"
    else:
        nn = []
        for author_str, yr in data:
            rn = find_ref(author_str, yr)
            if rn and rn in old2new:
                nn.append(str(old2new[rn]))
        rep = "[" + ",".join(nn) + "]" if nn else orig
    new_body = new_body[:pos] + rep + new_body[pos + len(orig):]

# Rebuild references in new order
new_refs = []
for old_num in sorted(old2new.keys(), key=lambda x: old2new[x]):
    new_refs.append(f"{old2new[old_num]}. {refs[old_num]}")

# Add uncited references at the end
for old_num in sorted(refs.keys()):
    if old_num not in old2new:
        new_refs.append(f"{nxt}. {refs[old_num]}")
        nxt += 1

final = new_body + ref_header + "\n\n".join(new_refs) + "\n"
with open(fp, "w", encoding="utf-8") as f:
    f.write(final)

print(f"Renumbered {len(old2new)} cited refs, total refs: {len(new_refs)}")

# Verify
with open(fp, "r", encoding="utf-8") as f:
    verify = f.read()

all_brackets = list(re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", verify.split("References")[0]))
all_parens = list(re.finditer(r"\([A-Za-z\-]+", verify.split("References")[0]))
print(f"\nRemaining bracket citations: {len(all_brackets)}")
print(f"Remaining parenthetical author citations: {len(all_parens)}")
if all_parens:
    for m in all_parens[:10]:
        print(f"  - {m.group(0)}...")

print("\nDone!")
