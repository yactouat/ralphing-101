#!/bin/bash

# Make sure we fail fast if something goes horribly wrong in bash
set -e

# Record start time for duration display
START_TIME=$(date +%s)

# Allow Ctrl+C to exit cleanly (show duration on interrupt)
trap 'echo ""; echo "Interrupted. Exiting Ralph loop."; ELAPSED=$(($(date +%s) - START_TIME)); echo "Total time: $((ELAPSED / 60))m $((ELAPSED % 60))s"; exit 130' INT TERM

echo "Starting the Ralph Wiggum loop. I'm helping!"

# Ensure out/ralph exists and our state file is there
mkdir -p out/ralph
STATE_FILE="out/ralph/progress.txt"
if [ ! -f "$STATE_FILE" ]; then
    echo "STATUS: IN_PROGRESS" > "$STATE_FILE"
fi

ITERATION=1

while true; do
    echo "========================================"
    echo "Starting Iteration $ITERATION"
    echo "========================================"

    # 1. Check for the termination signal
    if grep -q "STATUS: COMPLETE" "$STATE_FILE"; then
        echo "Task completed successfully! Exiting Ralph loop."
        ELAPSED=$(($(date +%s) - START_TIME))
        echo "Total time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
        break
    fi

    # 2. Invoke Claude Code with a strict, autonomous prompt
    # --dangerously-skip-permissions is required so the subprocess can write files/run
    # commands without waiting for interactive approval (which would hang the loop).
    claude --dangerously-skip-permissions -p "You are an autonomous agent executing the Ralph Wiggum technique.
    1. Read PRD.md to understand the end goal.
    2. Read out/ralph/progress.txt to see the current state.
    3. Look at the codebase (in out/ralph EXCLUSIVELY) to verify the actual state.
    4. Execute ONLY the very next logical step (write code, fix a bug, or write a test).
    5. Run the tests.
    6. If the step is done, update out/ralph/progress.txt with what was completed and what is next.
    7. If ALL requirements in PRD.md are met and all tests pass, replace 'STATUS: IN_PROGRESS' with 'STATUS: COMPLETE' in out/ralph/progress.txt.
    Do not ask for user confirmation. Just execute the step and exit."
    
    # 3. Prevent aggressive API rate limiting or infinite rapid-fire loops if it fails instantly
    echo "Iteration $ITERATION finished. Sleeping for 5 seconds..."
    sleep 5
    
    ((ITERATION++))
done