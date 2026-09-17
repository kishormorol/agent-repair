# Overleaf ↔ GitHub sync

Project: `6aabd6e1c7ddbb2652123f49` ("ICLR"), currently holding only a stub
`main.tex`. The account is on **Overleaf Premium**, so both Git access and
GitHub Sync are available.

The upload bundle is ready at `output/overleaf/agent-repair-paper.zip`
(70 files, 0.3 MB): `iclr2027.tex`, the conference style and bibliography,
`generated/` and `tables/`. PNG duplicates and the build directory are excluded
because Overleaf compiles the PDF figures.

## Why this needs you rather than me

Both routes end at an authentication step I will not perform: a Git token or a
GitHub OAuth authorization. Entering credentials on your behalf is not something
I do, even with your permission, so the steps below are yours. Everything either
route consumes is already prepared and committed.

---

## Route A — GitHub Sync (recommended: this is the actual two-way sync)

In the Overleaf project: **Menu → Sync → GitHub**.

1. Authorize Overleaf to access GitHub when prompted.
2. Choose **link to an existing repository** and select
   `kishormorol/agent-repair`, branch `close-review-gaps`.
3. Then **Menu → Settings → Main document** and set it to
   `paper/iclr2027.tex`. Overleaf will not find it otherwise, because the paper
   is not at the repository root.

### What to expect

- **Sync is manual, not continuous.** You press *Sync* to pull changes in or
  push Overleaf edits out. Nothing happens automatically on save.
- **Overleaf receives the whole repository** — code, tests and outputs, about
  7 MB packed. That is fine for size, but the file tree will be busier than a
  paper-only project.
- **Overleaf edits become commits** on the branch when you push, so my work and
  yours meet as ordinary git history and merge the usual way.

### The one thing to be careful about

`paper/generated/` is **machine-written**. Editing those files in Overleaf will
work until the next `make iclr-draft`, which overwrites them. Edit
`paper/iclr2027.tex` and let the generators own everything under `generated/`.
If a number looks wrong, the fix belongs in the script that emits it — that is
the property which caught a miscounted figure earlier.

---

## Route B — Overleaf Git remote (simpler, one project only)

Overleaf exposes each project as a git remote. From the repo:

```bash
git remote add overleaf https://git.overleaf.com/6aabd6e1c7ddbb2652123f49
git subtree push --prefix=paper overleaf master     # first push
```

You will be asked for an Overleaf Git token — generate it under **Account →
Settings → Git integration**.

`git subtree` pushes only `paper/`, so Overleaf gets a clean paper-only project
with `iclr2027.tex` at the root and no main-document setting needed. The cost is
that pulling Overleaf edits back is a `git subtree pull`, which is clumsier than
Route A.

---

## Route C — if you just want the files there now

Upload `output/overleaf/agent-repair-paper.zip` through the Overleaf file tree's
upload button. Overleaf extracts zips. This gets everyone editing in minutes but
establishes **no sync at all**, so the Overleaf copy and the repository will
drift apart. Use it only as a stopgap.

---

## Recommendation

**Route A.** It is what you asked for, it is the only one that syncs both ways,
and the repository is already well within Overleaf's limits. Delete the stub
`main.tex` after the first sync so the project has one obvious main document.

Tell me once the link exists and I will verify the Overleaf copy compiles to the
same 40-page, 9-page-main-text PDF the repository produces, so we know the two
sides genuinely agree before anyone starts editing.
