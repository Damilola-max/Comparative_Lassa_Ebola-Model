#!/usr/bin/env python3
import re

FP = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/FULL_MANUSCRIPT.md"

text = open(FP, encoding="utf-8").read()
if "References" not in text:
    raise SystemExit("No References section found")

body, refs_text = text.split("References", 1)

refs = {
    int(n): e.strip()
    for n, e in re.findall(r"^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)", refs_text, re.M | re.S)
}

old_to_new = {}
next_num = 1
for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
    for x in m.group(1).split(","):
        old = int(x.strip())
        if old not in old_to_new:
            old_to_new[old] = next_num
            next_num += 1

new_body = re.sub(
    r"\[(\d+(?:\s*,\s*\d+)*)\]",
    lambda m: "[" + ",".join(str(old_to_new[int(x.strip())]) for x in m.group(1).split(",")) + "]",
    body,
)

ordered_refs = []
for old, new in sorted(old_to_new.items(), key=lambda kv: kv[1]):
    if old in refs:
        ordered_refs.append((new, refs[old]))

for old in sorted(refs):
    if old not in old_to_new:
        ordered_refs.append((next_num, refs[old]))
        next_num += 1

new_text = new_body + "References\n" + "\n\n".join(f"{n}. {e}" for n, e in ordered_refs) + "\n"
open(FP, "w", encoding="utf-8").write(new_text)

# verify
body2, refs2 = new_text.split("References", 1)
groups = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body2)
first = int(groups[0].split(",")[0].strip()) if groups else None
ref_nums = [int(n) for n in re.findall(r"^\s*(\d+)\.", refs2, re.M)]

print("first_intext:", first)
print("references_count:", len(ref_nums))
print("references_sequential:", ref_nums == list(range(1, len(ref_nums) + 1)))
