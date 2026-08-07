# VHS tape — wrap refuses a configuration that cannot work.
#   render:  vhs docs/tapes/outage-guard.tape
# This is the P1 that took Claude Code down three times in one day: a base URL
# pinned in settings.json outranks the environment `distil wrap` sets, so when
# the pinned service is not running EVERY session fails with an error that
# blames the provider.

Output docs/assets/outage-guard.gif

Set Shell "bash"
Set FontSize 17
Set Width 1100
Set Height 520
Set Theme "Catppuccin Mocha"
Set Padding 22
Set TypingSpeed 55ms

Type "# settings.json pins a port whose service is gone"
Enter
Type "cat .claude/settings.json"
Enter
Sleep 2s

Type "distil wrap -- claude -p 'hi'"
Enter
Sleep 3s

Type "# refused BEFORE launching, with the fix named"
Enter
Sleep 2500ms
