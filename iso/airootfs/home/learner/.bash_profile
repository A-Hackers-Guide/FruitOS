# On the first console, FruitOS starts straight into the learning app.
# Quit the app (Alt+F4 / Ctrl+Q) to get a shell; type `startfruit` to reopen it.
alias startfruit='cage -- /usr/local/bin/fruitos-session'
if [[ -z $WAYLAND_DISPLAY && $(tty) == /dev/tty1 ]]; then
  cage -- /usr/local/bin/fruitos-session
fi
