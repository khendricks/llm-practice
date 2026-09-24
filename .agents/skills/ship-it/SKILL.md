---
name: ship-it
description: Validate, commit, and push completed changes to this personal project's main branch. Use when the user asks to ship, commit, or push current work.
---

# Ship It

Publish the user's completed, in-scope changes with a clean, traceable Git
history.

1. Inspect the current branch, remote, working tree, and diff. Confirm the
   files being shipped match the user's requested work. Do not stage
   unrelated changes or credentials.
2. Run the repository's standard test command: ``uv run python -m pytest
   -v``. Stop and report test failures rather than publishing a failing
   change, unless the user explicitly directs otherwise.
3. Stage only the intended files, review the staged diff, and create a
   concise imperative commit message that describes the change.
4. Push the commit to ``origin/main``. This is a personal project, and the
   user has approved direct pushes to ``main``. If the user requests a pull
   request instead, use a feature branch and open one.
5. Verify the push or pull-request creation succeeded. Report the commit
   hash and the branch or pull-request URL.

If the working tree has conflicts, the remote rejects the push, the target
branch is ambiguous, or publishing would include unrelated work, pause and
ask the user how to proceed. Do not force-push unless they explicitly ask.
