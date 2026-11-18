# Activate the preview environment and run from the repository root.
# Render with vhs docs/sdpa-preview.tape (requires ttyd and ffmpeg).
Output docs/sdpa-preview.gif
Set Shell "bash"
Set Theme "Dracula"
Set FontFamily "DejaVu Sans Mono"
Set FontSize 15
Set Width 1540
Set Height 850
Set Padding 20
Set Framerate 12

Type "python scripts/demo_sdpa.py"
Enter
Sleep 15s
