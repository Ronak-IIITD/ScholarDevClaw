# nanogpt-opt demo: the lean 4-command workflow on a real external repo.
# Render with: vhs demo-nanogpt.tape   (requires vhs + ttyd + ffmpeg)
Output demo-nanogpt.mp4
Set Theme Dracula
Set FontSize 17
Set Width 1180
Set Height 740
Set LineHeight 1.25

# 1) analyze — offline, no LLM, only frozen 10-spec scope
Type "nanogpt-opt analyze test_repos/external-nanogpt"
Enter
Sleep 2.5s

# 2) suggest — map one spec to concrete file:line targets with confidence
Type "nanogpt-opt suggest test_repos/external-nanogpt --spec rmsnorm"
Enter
Sleep 2.5s

# 3) apply (dry-run default) — unified diff, zero writes
Type "nanogpt-opt apply test_repos/external-nanogpt --spec rmsnorm"
Enter
Sleep 2.5s

# 4) apply --write — atomically-per-spec with revert manifest
Type "nanogpt-opt apply test_repos/external-nanogpt --spec rmsnorm --write"
Enter
Sleep 3s

# 5) second write — honest idempotency: nothing to apply
Type "nanogpt-opt apply test_repos/external-nanogpt --spec rmsnorm --write"
Enter
Sleep 2s

# 6) validate — honest scorecard: a stage that did not run is skipped, never a pass
Type "nanogpt-opt validate test_repos/external-nanogpt; echo exit=$?"
Enter
Sleep 5s

# 7) show what actually changed
Type "git -C test_repos/external-nanogpt status --short"
Enter
Sleep 2s

# 8) revert — byte-identical restore, manifest removed
Type "nanogpt-opt apply test_repos/external-nanogpt --revert"
Enter
Sleep 2s

# 9) proof: clean git status again
Type "git -C test_repos/external-nanogpt status --short && echo CLEAN-revert"
Enter
Sleep 4s
