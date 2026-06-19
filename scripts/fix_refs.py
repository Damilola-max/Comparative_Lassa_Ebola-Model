#!/usr/bin/env python3
import re

fp = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/FULL_MANUSCRIPT.md"
with open(fp, "r", encoding="utf-8") as f:
    text = f.read()

# Split body and references - handle plain text "References" header
m = re.search(r"(^References\s*\n)", text, re.MULTILINE)
if not m:
    m = re.search(r"(## References\s*\n)", text)
    if not m:
        print("ERROR: No References section found")
        exit(1)

body = text[:m.start()]
ref_header = m.group(1)
ref_text = text[m.end():]

# Parse references: number -> text (handle indented numbers like "    1. ")
refs = {}
for num, entry in re.findall(r"^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)", ref_text, re.MULTILINE | re.DOTALL):
    refs[int(num)] = entry.strip()

print(f"Found {len(refs)} references")

# Build flexible author-year lookup: for each ref, store first author string and year
ref_lookup = []
for num, entry in refs.items():
    first_line = entry.split("\n")[0]
    # Extract everything before first comma as first author full name
    author_part = first_line.split(",")[0] if "," in first_line else first_line.split(".")[0]
    # Extract year
    ym = re.search(r"\((\d{4})[a-z]?\)", first_line)
    yr = ym.group(1) if ym else ""
    ref_lookup.append((num, author_part.strip(), yr, entry))

def find_ref_by_author_year(author_query, year_query):
    """Find reference number matching author_query and year."""
    # Try exact match on first author string
    for num, author_str, yr, _ in ref_lookup:
        if yr == year_query and author_query.lower() in author_str.lower():
            return num
    # Try matching last word of author_query against author_str
    last_word = author_query.split()[-1] if " " in author_query else author_query
    for num, author_str, yr, _ in ref_lookup:
        if yr == year_query and last_word.lower() in author_str.lower():
            return num
    # Try fuzzy: check if any word in author_query matches
    words = author_query.split()
    for num, author_str, yr, _ in ref_lookup:
        if yr == year_query:
            for w in words:
                if len(w) > 2 and w.lower() in author_str.lower():
                    return num
    return None

# Citation finder
citations = []
# numbered [1] or [1,2]
for mo in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
    citations.append((mo.start(), "num", [int(n.strip()) for n in mo.group(1).split(",")], mo.group(0)))

# author-year (Author et al., 2020; Author2 et al., 2021)
for mo in re.finditer(r"\(([A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4}(?:;\s*[A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4})*)\)", body):
    inner = mo.group(1)
    ind = re.findall(r"([A-Za-z\-\s]+?)\s+et\s+al\.\s*,\s*(\d{4})", inner)
    citations.append((mo.start(), "ay", [(a.strip(), y) for a, y in ind], mo.group(0)))

# simple (Author, 2020) - also handle single surname like (Di, 2022) or (Sims, 2025)
for mo in re.finditer(r"\(([A-Za-z\-\s]+?),\s*(\d{4})\)", body):
    author_str, yr = mo.group(1).strip(), mo.group(2)
    rn = find_ref_by_author_year(author_str, yr)
    if rn:
        citations.append((mo.start(), "simple", [(author_str, yr)], mo.group(0)))

citations.sort(key=lambda x: x[0])
print(f"Found {len(citations)} citation groups")

# Map old numbers to sequential
old2new = {}
nxt = 1
for pos, ctype, data, orig in citations:
    if ctype == "num":
        for old in data:
            if old not in old2new:
                old2new[old] = nxt
                nxt += 1
    else:
        for author_str, yr in data:
            rn = find_ref_by_author_year(author_str, yr)
            if rn and rn not in old2new:
                old2new[rn] = nxt
                nxt += 1
            elif rn is None:
                print(f"WARNING: Could not find ref for '{author_str}', {yr}")

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
            rn = find_ref_by_author_year(author_str, yr)
            if rn and rn in old2new:
                nn.append(str(old2new[rn]))
        rep = "[" + ",".join(nn) + "]" if nn else orig
    new_body = new_body[:pos] + rep + new_body[pos + len(orig):]

# Rebuild references
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
