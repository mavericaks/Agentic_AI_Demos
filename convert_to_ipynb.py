import os
import glob
import json
import re

def convert_demo(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    
    # 1. Extract docstring to markdown cell
    doc_match = re.match(r'^"""\n?(.*?)\n?"""\n', content, re.DOTALL)
    if doc_match:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in doc_match.group(1).split("\n")]})
        content = content[doc_match.end():]

    # 2. Remove __main__ block
    content = re.sub(r'if __name__ == "__main__":\n\s+\w+\(\)\n?', '', content)

    # 3. Unindent the main run_xxx() function
    func_match = re.search(r'def run_\w+\(\):\n(.*)', content, re.DOTALL)
    if func_match:
        body = func_match.group(1)
        # Remove print headers like `print("=" * 60)` at the start of functions
        body = re.sub(r'^\s*print\("=" \* 60\)\n\s*print\("  SESSION.*?"\)\n\s*print\("=" \* 60\)\n', '', body, flags=re.MULTILINE)
        
        # unindent 4 spaces and replace stray returns
        unindented_lines = []
        for line in body.split("\n"):
            stripped = line[4:] if line.startswith("    ") else line
            # Replace 'return' at the top level with a safe break/pass
            if stripped.strip() == "return":
                stripped = stripped.replace("return", "raise RuntimeError('Stop Execution')")
            unindented_lines.append(stripped)
            
        unindented = "\n".join(unindented_lines)
        content = content[:func_match.start()] + unindented

    # 4. Split by large block comments or step headers
    # We remove DOTALL for the single-line headers to prevent catastrophic gobbling
    split_pattern = r'(?:# ═{10,}\n(.*?)# ═{10,}\n|# ─{2,} ([^\n]*?) ─{2,}\n)'
    blocks = re.split(split_pattern, content, flags=re.DOTALL)
    
    # blocks structure from re.split with 2 capture groups:
    # [ text, match1, match2, text, match1, match2, text ... ]
    # One of match1 or match2 will be None.
    
    if blocks[0].strip():
        cells.append({"cell_type": "code", "metadata": {}, "source": [line + "\n" for line in blocks[0].strip().split("\n")], "outputs": [], "execution_count": None})
        
    for i in range(1, len(blocks), 3):
        match1 = blocks[i]
        match2 = blocks[i+1]
        code_block = blocks[i+2].strip()
        
        raw_header = match1 if match1 is not None else match2
        
        # Clean header
        header_lines = [line.strip().lstrip('#').strip() for line in raw_header.strip().split('\n')]
        header_text = "### " + header_lines[0]
        if len(header_lines) > 1:
            header_text += "\n" + "\n".join(header_lines[1:])
            
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in header_text.split("\n")]})
        
        if code_block:
            cells.append({"cell_type": "code", "metadata": {}, "source": [line + "\n" for line in code_block.split("\n")], "outputs": [], "execution_count": None})
            
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}
    
    # Cleanup trailing newlines in source arrays
    for c in nb['cells']:
        if c['source'] and c['source'][-1].endswith('\n'):
            c['source'][-1] = c['source'][-1].rstrip('\n')
            
    out_path = filepath.replace(".py", ".ipynb")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Created {out_path}")

for f in glob.glob("session_*/*.py"):
    if "mcp_server" in f: continue
    convert_demo(f)
