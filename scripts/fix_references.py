#!/usr/bin/env python3
"""
Script to renumber manuscript references sequentially by first appearance.
Handles both numbered [X] and author-year (Author et al., YYYY) citations.
"""

import re
import sys

def extract_references(text):
    """Extract reference entries from the References section."""
    # Find References section
    ref_match = re.search(r'## References\s*\n(.*)', text, re.DOTALL)
    if not ref_match:
        return {}
    
    ref_text = ref_match.group(1)
    # Split into individual references - look for numbered entries
    ref_entries = re.findall(r'^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s|\Z)', ref_text, re.MULTILINE | re.DOTALL)
    
    refs = {}
    for num, entry in ref_entries:
        entry = entry.strip()
        # Extract first author surname
        author_match = re.match(r'^([A-Za-z\-\s\.]+?),', entry)
        if author_match:
            first_author = author_match.group(1).strip().split()[-1]  # Get surname
            year_match = re.search(r'\((\d{4})[a-z]?\)', entry)
            year = year_match.group(1) if year_match else ""
            key = f"{first_author}_{year}"
            refs[int(num)] = {
                'text': entry,
                'first_author': first_author,
                'year': year,
                'key': key
            }
    return refs

def find_author_year_citations(text):
    """Find all author-year format citations in text."""
    # Pattern: (Author et al., YYYY; Author2 et al., YYYY) or (Author, YYYY)
    pattern = r'\(([A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4}(?:;\s*[A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4})*)\)'
    matches = re.finditer(pattern, text)
    citations = []
    for m in matches:
        # Parse individual citations within the group
        inner = m.group(1)
        individual = re.findall(r'([A-Za-z\-\s]+?)\s+et\s+al\.\s*,\s*(\d{4})', inner)
        citations.append((m.start(), m.end(), m.group(0), individual))
    return citations

def find_numbered_citations(text):
    """Find all numbered citations [X] or [X,Y] in text."""
    pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
    matches = re.finditer(pattern, text)
    citations = []
    for m in matches:
        nums = [int(n.strip()) for n in m.group(1).split(',')]
        citations.append((m.start(), m.end(), m.group(0), nums))
    return citations

def build_author_to_ref_map(refs):
    """Build mapping from author surname to reference number."""
    mapping = {}
    for num, ref in refs.items():
        surname = ref['first_author']
        year = ref['year']
        # Handle multiple authors with same surname by using full key
        mapping[f"{surname}_{year}"] = num
        # Also add just surname for lookup
        if surname not in mapping:
            mapping[surname] = []
        mapping[surname].append((year, num))
    return mapping

def main():
    filepath = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/FULL_MANUSCRIPT.md"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into body and references
    ref_section_match = re.search(r'(## References\s*\n)', text)
    if not ref_section_match:
        print("No References section found")
        sys.exit(1)
    
    body = text[:ref_section_match.start()]
    ref_header = ref_section_match.group(1)
    ref_text = text[ref_section_match.end():]
    
    # Extract references
    refs = {}
    ref_entries = re.findall(r'^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)', ref_text, re.MULTILINE | re.DOTALL)
    for num, entry in ref_entries:
        entry = entry.strip()
        refs[int(num)] = entry
    
    print(f"Found {len(refs)} references")
    
    # Build author lookup from references
    author_map = {}
    for num, entry in refs.items():
        # Extract first author surname - handle various formats
        lines = entry.split('\n')
        first_line = lines[0] if lines else entry
        
        # Pattern: Surname, F. M. or Surname, F.M.
        author_match = re.match(r'^([A-Za-z\-]+?),', first_line)
        if author_match:
            surname = author_match.group(1).strip()
            year_match = re.search(r'\((\d{4})[a-z]?\)', first_line)
            if year_match:
                year = year_match.group(1)
                author_map[f"{surname}_{year}"] = num
                if surname not in author_map:
                    author_map[surname] = []
                author_map[surname].append((year, num))
    
    # Find all citations in body
    all_citations = []
    
    # Numbered citations
    for m in re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', body):
        nums = [int(n.strip()) for n in m.group(1).split(',')]
        all_citations.append((m.start(), 'numbered', nums, m.group(0)))
    
    # Author-year citations - handle (Author et al., YYYY; Author2 et al., YYYY)
    for m in re.finditer(r'\(([A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4}(?:;\s*[A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4})*)\)', body):
        inner = m.group(1)
        individual = re.findall(r'([A-Za-z\-\s]+?)\s+et\s+al\.\s*,\s*(\d{4})', inner)
        parsed = []
        for auth, yr in individual:
            surname = auth.strip().split()[-1]
            parsed.append((surname, yr))
        all_citations.append((m.start(), 'author_year', parsed, m.group(0)))
    
    # Also handle simple (Author, YYYY) format
    for m in re.finditer(r'\(([A-Za-z\-]+?),\s*(\d{4})\)', body):
        surname = m.group(1).strip()
        year = m.group(2)
        # Check if this looks like a citation (surname should be in our refs)
        if surname in author_map or f"{surname}_{year}" in author_map:
            all_citations.append((m.start(), 'simple_author_year', [(surname, year)], m.group(0)))
    
    # Sort by position
    all_citations.sort(key=lambda x: x[0])
    
    # Build sequential mapping
    old_to_new = {}
    next_num = 1
    
    for pos, ctype, data, original in all_citations:
        if ctype == 'numbered':
            for old_num in data:
                if old_num not in old_to_new:
                    old_to_new[old_num] = next_num
                    next_num += 1
        elif ctype == 'author_year' or ctype == 'simple_author_year':
            for surname, year in data:
                key = f"{surname}_{year}"
                # Find corresponding ref number
                ref_num = None
                if key in author_map:
                    ref_num = author_map[key]
                elif surname in author_map and isinstance(author_map[surname], list):
                    # Find matching year
                    for yr, num in author_map[surname]:
                        if yr == year:
                            ref_num = num
                            break
                
                if ref_num and ref_num not in old_to_new:
                    old_to_new[ref_num] = next_num
                    next_num += 1
                elif ref_num and ref_num in old_to_new:
                    pass  # Already mapped
                else:
                    print(f"WARNING: Could not find reference for {surname} et al., {year}")
    
    print(f"Mapped {len(old_to_new)} unique references")
    
    # Replace citations in body
    new_body = body
    
    # Replace numbered citations - need to work backwards to preserve positions
    numbered_matches = list(re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', new_body))
    for m in reversed(numbered_matches):
        nums = [int(n.strip()) for n in m.group(1).split(',')]
        new_nums = [str(old_to_new.get(n, n)) for n in nums]
        replacement = '[' + ','.join(new_nums) + ']'
        new_body = new_body[:m.start()] + replacement + new_body[m.end():]
    
    # Replace author-year citations
    # (Author et al., YYYY; Author2 et al., YYYY) -> [X,Y]
    for m in reversed(list(re.finditer(r'\(([A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4}(?:;\s*[A-Za-z\-\s]+?\s+et\s+al\.\s*,\s*\d{4})*)\)', new_body))):
        inner = m.group(1)
        individual = re.findall(r'([A-Za-z\-\s]+?)\s+et\s+al\.\s*,\s*(\d{4})', inner)
        new_nums = []
        for auth, yr in individual:
            surname = auth.strip().split()[-1]
            key = f"{surname}_{yr}"
            ref_num = None
            if key in author_map:
                ref_num = author_map[key]
            elif surname in author_map and isinstance(author_map[surname], list):
                for y, n in author_map[surname]:
                    if y == yr:
                        ref_num = n
                        break
            if ref_num and ref_num in old_to_new:
                new_nums.append(str(old_to_new[ref_num]))
        if new_nums:
            replacement = '[' + ','.join(new_nums) + ']'
            new_body = new_body[:m.start()] + replacement + new_body[m.end():]
    
    # Replace simple (Author, YYYY) citations
    for m in reversed(list(re.finditer(r'\(([A-Za-z\-]+?),\s*(\d{4})\)', new_body))):
        surname = m.group(1).strip()
        year = m.group(2)
        key = f"{surname}_{year}"
        ref_num = None
        if key in author_map:
            ref_num = author_map[key]
        elif surname in author_map and isinstance(author_map[surname], list):
            for y, n in author_map[surname]:
                if y == year:
                    ref_num = n
                    break
        if ref_num and ref_num in old_to_new:
            replacement = '[' + str(old_to_new[ref_num]) + ']'
            new_body = new_body[:m.start()] + replacement + new_body[m.end():]
    
    # Rebuild references section
    new_refs = []
    for old_num in sorted(old_to_new.keys(), key=lambda x: old_to_new[x]):
        new_num = old_to_new[old_num]
        ref_text_entry = refs.get(old_num, "")
        new_refs.append(f"{new_num}. {ref_text_entry}")
    
    # Add any references not cited
    cited_nums = set(old_to_new.keys())
    for old_num in sorted(refs.keys()):
        if old_num not in cited_nums:
            new_refs.append(f"{next_num}. {refs[old_num]}")
            next_num += 1
    
    new_ref_section = ref_header + '\n\n'.join(new_refs) + '\n'
    
    # Combine
    final_text = new_body + new_ref_section
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print("Done! References renumbered sequentially.")
    print(f"New reference count: {len(new_refs)}")

if __name__ == '__main__':
    main()
