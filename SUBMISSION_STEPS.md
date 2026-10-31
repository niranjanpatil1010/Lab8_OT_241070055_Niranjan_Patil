# What to do before you submit (about 30 minutes)

## 1. Put the raw chat exports in `prompts/exports/` (see the README there). Remove secrets (API keys, tokens, passwords).
## 2. Fill the two things only you know
- README → AI tools register: the exact model Antigravity used per track (it is recorded only approximately).
- Check CHECKLIST.md against your class notes; add any topic your teacher taught that is missing.
## 3. Push to a PUBLIC GitHub repo
```
# create an empty public repo on github.com (no README/license), e.g. optimize-anything, then:
cd optimize-anything                    # the unzipped folder (it already has a .git with the commits)
git status                              # should be clean
git log --oneline                       # check the commits are there
git remote add origin https://github.com/<your-username>/optimize-anything.git
git branch -M main
git push -u origin main
```
- If git asks for a password: use a Personal Access Token (GitHub → Settings → Developer settings → Tokens), not your password; or use GitHub Desktop / `gh auth login`.
- If `git commit` ever complains about identity: `git config user.name "Your Name"` and `git config user.email "you@example.com"`.
- Your own edits (exports, README fixes): `git add -A && git commit -m "[manual] add chat exports and model versions" && git push`.
- Open the repo in a private/incognito window to confirm it is public and the README shows.
## 4. Run it once on your machine (shows reproducibility)
```
pip install numpy scipy matplotlib pytest pennylane scikit-learn coverage
python3 trackJ/profile_suite.py
python3 -m pytest tests -q          # expect all passed (Track G tests need QEMU, otherwise skipped)
```
## 5. Record the demo (5 minutes) — follow DEMO_SCRIPT.md
## 6. Viva — read VIVA_NOTES.md; open each file once and be able to explain it without AI.
