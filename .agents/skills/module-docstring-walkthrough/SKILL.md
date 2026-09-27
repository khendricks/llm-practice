---
name: module-docstring-walkthrough
description: Add a concise conceptual walkthrough to a source-file module
  docstring, using ASCII data-flow diagrams when they clarify the logic. Use
  when documenting a code flow; do not use for executable tutorials or API
  docs.
---

# Module Docstring Walkthrough

Add or update the module-level docstring without changing the implementation.

Describe the file's actual logic in its natural order:

1. State the file's purpose and identify the input and output.
2. Walk through meaningful transformations or decisions in sequence.
3. Explain how intermediate results become the final result.

When a pipeline, matrix, state transition, or hierarchy benefits from a visual,
include a compact indented ASCII representation of the data at each important
phase. Normally use descriptive symbolic labels drawn from the code. When the
user asks for exact example values, instead use a small, clearly labeled
illustrative input and show the resulting values at each phase. State any
simplifying assumptions, such as fixed weights, and do not present them as the
program's runtime values. When code has separate learned transforms, use
distinct illustrative matrices unless the user specifically requests a
simplified shared projection. Skip diagrams when they would only restate
simple prose.

Keep the walkthrough concise, accurate, and explanatory. Do not use Python
prompts, doctests, or code intended to be executed. Preserve the module's
existing style and avoid altering runtime behavior.
