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

# Parse references: number -> text
refs = {}
for num, entry in re.findall(r"^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)", ref_text, re.MULTILINE | re.DOTALL):
    refs[int(num)] = entry.strip()

print(f"Found {len(refs)} references")

# Build reference lookup: for each ref, store first author and year
ref_lookup = []
for num, entry in refs.items():
    first_line = entry.split("\n")[0]
    # Extract first author(s) before first comma
    author_part = first_line.split(",")[0] if "," in first_line else first_line.split(".")[0]
    # Extract year - try multiple patterns
    yr = ""
    ym = re.search(r"\((\d{4})[a-z]?\)", first_line)
    if ym:
        yr = ym.group(1)
    else:
        ym2 = re.search(r",\s*(\d{4})[a-z]?\s*[\.\)]", first_line)
        if ym2:
            yr = ym2.group(1)
        else:
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
            if author_query.lower() in author_str.lower() or author_str.lower() in author_query.lower():
                return num
            candidates.append((num, author_str))
    aq_norm = normalize(author_query)
    for num, author_str in candidates:
        as_norm = normalize(author_str)
        if aq_norm in as_norm or as_norm in aq_norm:
            return num
        aq_words = set(author_query.lower().split())
        as_words = set(author_str.lower().split())
        if aq_words & as_words:
            return num
    for num, author_str in candidates:
        as_norm = normalize(author_str)
        if abs(len(aq_norm) - len(as_norm)) <= 2:
            min_len = min(len(aq_norm), len(as_norm))
            matches = sum(1 for i in range(min_len) if i < len(aq_norm) and i < len(as_norm) and aq_norm[i] == as_norm[i])
            if matches >= min_len - 2:
                return num
    return None

# Extract author-year pairs from a citation group string
def extract_citations(group_text):
    """Extract list of (author, year) tuples from citation group text."""
    # Pattern for individual citations within a group:
    # - Author et al., YYYY
    # - Author & Author2, YYYY
    # - Author, YYYY
    # - Author and Author2, YYYY
    pattern = r"([A-Za-z\-\s]+?(?:\s+(?:et\s+al\.|&\s*[A-Za-z\-\s]+|and\s+[A-Za-z\-\s]+))?)\s*,\s*(\d{4})"
    results = []
    for mo in re.finditer(pattern, group_text):
        author_str = mo.group(1).strip()
        yr = mo.group(2)
        # Skip if author is empty or just whitespace
        if author_str and not re.match(r"^\s*\d+\s*$", author_str):
            results.append((author_str, yr))
    return results

# Find all citation groups in body
citations = []

# Find bracketed groups containing years: [something with 4 digits]
for mo in re.finditer(r"\[([^\]]*\d{4}[^\]]*)\]", body):
    inner = mo.group(1)
    pairs = extract_citations(inner)
    if pairs:
        # Only include if at least one pair matches a reference
        matched = [p for p in pairs if find_ref(p[0], p[1])]
        if matched:
            citations.append((mo.start(), "bracket", matched, mo.group(0)))

# Find parenthetical groups containing years: (something with 4 digits)
for mo in re.finditer(r"\(([^)]*\d{4}[^)]*)\)", body):
    inner = mo.group(1)
    pairs = extract_citations(inner)
    if pairs:
        matched = [p for p in pairs if find_ref(p[0], p[1])]
        if matched:
            citations.append((mo.start(), "paren", matched, mo.group(0)))

# Remove overlapping/duplicate citations (if a group is inside another)
citations.sort(key=lambda x: x[0])
filtered = []
for i, c in enumerate(citations):
    pos, ctype, data, orig = c
    # Skip if this is fully contained within a previous citation
    if filtered:
        prev_pos, prev_ctype, prev_data, prev_orig = filtered[-1]
        if pos >= prev_pos and pos + len(orig) <= prev_pos + len(prev_orig):
            continue
    filtered.append(c)
citations = filtered

print(f"Found {len(citations)} citation groups")

# Map old ref numbers to sequential
old2new = {}
nxt = 1
unmatched = []

for pos, ctype, data, orig in citations:
    for author_str, yr in data:
        rn = find_ref(author_str, yr)
        if rn and rn not in old2new:
            old2new[rn] = nxt
            nxt += 1
        elif rn is None:
            unmatched.append((author_str, yr, orig))

if unmatched:
    seen = set()
    print(f"\nWARNING: {len(unmatched)} unmatched citations:")
    for a, y, o in unmatched:
        key = (a, y)
        if key not in seen:
            seen.add(key)
            print(f"  - '{a}', {y}")

print(f"Mapped {len(old2new)} unique cited references")

# Replace citations in body (backwards to preserve positions)
new_body = body
for pos, ctype, data, orig in reversed(citations):
    nn = []
    for author_str, yr in data:
        rn = find_ref(author_str, yr)
        if rn and rn in old2new:
            nn.append(str(old2new[rn]))
    rep = "[" + ",".join(nn) + "]" if nn else orig
    new_body = new_body[:pos] + rep + new_body[pos + len(orig):]

# Also handle pure numeric citations [N] or [N,M] that might reference old numbers
# We need to remap them if they reference old ref numbers
numeric_citations = []
for mo in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", new_body):
    nums = [int(n.strip()) for n in mo.group(1).split(",")]
    # Check if these numbers correspond to old reference numbers
    if all(n in old2new for n in nums):
        numeric_citations.append((mo.start(), nums, mo.group(0)))

# Replace numeric citations (backwards)
for pos, nums, orig in reversed(numeric_citations):
    nn = [str(old2new[n]) for n in nums]
    rep = "[" + ",".join(nn) + "]"
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

vbody = verify.split("References")[0]
all_parens = re.findall(r"\([A-Za-z\-]+", vbody)
print(f"\nRemaining parenthetical author citations: {len(all_parens)}")
if all_parens:
    for p in all_parens[:10]:
        print(f"  - {p}...")

# Check max citation number
nums = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", vbody)
all_nums = []
for n in nums:
    all_nums.extend([int(x.strip()) for x in n.split(",")])
if all_nums:
    print(f"Max citation number: {max(all_nums)}")
    print(f"Unique citation numbers: {len(set(all_nums))}")

print("\nDone!")
