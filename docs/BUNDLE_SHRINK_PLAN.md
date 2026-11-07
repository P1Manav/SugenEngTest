# Standby Plan: Reducing Git Bundle Size via `git filter-repo`

> [!IMPORTANT]
> **Status:** Documented plan only. **Do NOT run without explicit user authorization.**
> History rewriting alters all commit hashes from the first filtered commit forward.
> Use this procedure ONLY if Sugen rejects external download delivery and requires an email attachment under their size limit.

---

## 1. Root Cause of Repository & Bundle Bloat

Current repository measurements:
- `git count-objects -vH`: **581.36 MiB**
- `git bundle create test.bundle --all`: **551.87 MiB**

Although `models/` is no longer in the active working tree or `.git/index` (models are downloaded at `docker build` time), Git history retains all objects ever committed:
1. `models/paraphrase-multilingual-MiniLM-L12-v2/model.safetensors.part[0-4]`: ~470 MB across past commits.
2. `models/all-MiniLM-L6-v2/model.safetensors`: ~90 MB from the initial pipeline.
3. Historical versions of `kb/dense_embeddings.npz`: ~26 MB each.

Removing `models/` across Git history reduces the bundle size from **~552 MB to ~45 MB** (a >90% reduction), small enough to fit standard mail systems or cloud transfer.

---

## 2. Step-by-Step Execution Plan

### Step 1: Create a Safety Backup Clone
```bash
# Move to parent directory and create an exact backup clone
cd ..
git clone SugenEngTest SugenEngTest-prefilter-backup
cd SugenEngTest
```

### Step 2: Install `git-filter-repo`
```bash
pip install git-filter-repo
```

### Step 3: Run `git filter-repo` to Purge Historical Models
Filter out the `models/` directory entirely across all branches and tags:
```bash
git filter-repo --path models/ --invert-paths --force
```

### Step 4: Re-link Remote (filter-repo removes remotes for safety)
```bash
git remote add origin https://github.com/P1Manav/SugenEngTest.git
```

### Step 5: Remap Commit Hashes Cited in `DECISIONS.md`
`git-filter-repo` generates a mapping table of all rewritten commits at `.git/filter-repo/commit-map`.

Execute this automated remapping script to preserve documentation integrity:
```python
# scripts/remap_decisions_hashes.py
import re

commit_map = {}
with open(".git/filter-repo/commit-map", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            old_hash, new_hash = parts
            if old_hash != "0000000000000000000000000000000000000000":
                commit_map[old_hash[:7]] = new_hash[:7]
                commit_map[old_hash] = new_hash

with open("DECISIONS.md", "r", encoding="utf-8") as f:
    text = f.read()

for old_h, new_h in commit_map.items():
    text = re.sub(rf"\b{old_h}\b", new_h, text)

with open("DECISIONS.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully remapped all DECISIONS.md commit hashes.")
```

Commit the remapped hashes:
```bash
python scripts/remap_decisions_hashes.py
git commit -am "docs: remap DECISIONS.md commit hashes post-filter-repo"
```

### Step 6: Verify New Bundle Size
```bash
git count-objects -vH
git bundle create SugenEngTest_shrunk.bundle --all
ls -lh SugenEngTest_shrunk.bundle
```
*Expected result:* Bundle size **< 45 MB**.

### Step 7: Push to Remote (Requires Force Push)
```bash
git push origin master --force
```
