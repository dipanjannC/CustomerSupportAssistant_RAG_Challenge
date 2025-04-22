# Find listening ports to confirm no Phoenix conflicts:
sudo lsof -iTCP:4317 -sTCP:LISTEN
sudo lsof -iTCP:6006 -sTCP:LISTEN

# Graceful kill:
kill <PID>

# Forced kill:
sudo kill -9 <PID>

# Kill all by name:
sudo killall -9 process_name

# List processes with state:
ps -eo pid,ppid,stat,comm | grep <name_or_PID>

# Diagnose open files:
lsof -p <PID>

# Trace syscalls (Linux):
strace -p <PID>
