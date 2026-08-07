# VHS tape — the in-process library API.
#   render:  vhs docs/tapes/library-api.tape
# Checked in so the GIF is REPRODUCIBLE: a recording nobody can re-make goes
# stale silently and then misrepresents the product.

Output docs/assets/library-api.gif

Set Shell "bash"
Set FontSize 17
Set Width 1100
Set Height 620
Set Theme "Catppuccin Mocha"
Set Padding 22
Set TypingSpeed 55ms

Type "# distil as a library — no proxy, no daemon"
Enter
Sleep 900

Type "python3 - <<'PY'"
Enter
Type "from distil import compress_messages, expand_handle"
Enter
Type "log = '\n'.join(f'INFO worker heartbeat seq={i}' for i in range(400))"
Enter
Type "msgs = [{'role':'user','content':'why did it die?'},"
Enter
Type "        {'role':'tool','content':log}]"
Enter
Type "r = compress_messages(msgs)"
Enter
Type "print(f'{r.tokens_before} -> {r.tokens_after} tokens  ({r.saved_pct:.1f}% smaller)')"
Enter
Type "print('handles:', r.handles)"
Enter
Type "print('recovered byte-exact:', expand_handle(r.handles[0]) is not None)"
Enter
Type "PY"
Enter
Sleep 4s

Type "# nothing is gone — the original is one expand away"
Enter
Sleep 2500ms
