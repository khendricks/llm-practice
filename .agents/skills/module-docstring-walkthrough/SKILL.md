---
name: module-docstring-walkthrough
description: Add a concise conceptual walkthrough to a source-file module
  docstring, using ASCII data-flow diagrams when they clarify the logic. Use
  when documenting a code flow; do not use for executable tutorials or API
  docs.
---

# Module Docstring Walkthrough

Add or update the module-level docstring without changing the implementation.

Before writing, inspect closely related modules in the same feature family.
When a sibling module already has a walkthrough, reuse its conceptual order
and vocabulary, but not its length or incidental detail.

For a versioned module or a module with a suitable documented sibling, read
[the versioned walkthrough template](references/versioned-walkthrough.md)
before drafting. Use its section order and fill it with the target module's
actual behavior. Keep the version comparison to one or two plain-language
sentences.

Write for a beginner. Prefer common words, short sentences, and a small flow
diagram. For matrix or projection pipelines, use a small worked example with
real numbers, like the V1 attention walkthrough. Use two tokens and two
dimensions by default; use a larger example only when it explains something
important.

Describe the file's actual logic in its natural order:

1. State the file's purpose and identify the input and output.
2. Walk through meaningful transformations or decisions in sequence.
3. Explain how intermediate results become the final result.

Use compact ASCII diagrams to show the input and each meaningful intermediate
result. Choose fixed, reproducible example parameters rather than runtime
random values, and verify every displayed value from those parameters. State
that the values are illustrative. For a versioned sibling, state the
version-specific difference after the shared flow. Skip diagrams when they
would only restate simple prose.

Keep the walkthrough concise, accurate, and explanatory. Do not use Python
prompts, doctests, or code intended to be executed. Preserve the module's
existing style and avoid altering runtime behavior.
