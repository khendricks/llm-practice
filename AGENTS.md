# Project instructions

## Python style

- Follow PEP 8: use `lowercase_with_underscores` for functions and variables,
  `CapWords` for classes, and module-level `ALL_CAPS_WITH_UNDERSCORES` for
  constants. Do not use `l`, `O`, or `I` as variable names.
- Use four spaces, no tabs. Keep code lines to 79 characters and comments or
  docstrings to 72 characters. Wrap with parentheses or brackets, not
  backslashes.
- Group imports as standard library, third party, then local, with blank lines
  between groups. Do not use wildcard imports.
- Use two blank lines between top-level definitions and one between methods.
- Add docstrings to public classes, functions, and methods. Comments should
  explain why, rather than restate the code.
- Compare to `None` with `is` or `is not`; use `isinstance()` for type checks;
  catch specific exceptions; and do not assign a lambda to a name.
- Match a file's established style when it conflicts with these conventions.

## Test-driven development

For new behavior and bug fixes, work in a short red-green-refactor loop:

1. Write only enough of one test to make it fail.
2. Write only enough production code to make that test pass.
3. Refactor before moving to the next small behavior.

Add tests under the matching `tests/` subdirectory. Do not retrofit tests for
unrelated, existing code.

## Documentation

- Document only information that someone can realistically keep current.
- Do not duplicate code, type hints, docstrings, or another document; link or
  refer to the canonical source instead.
- Prefer intent, core logic, and verifiable references such as file paths,
  symbols, and commands over volatile implementation snapshots.
- When changing or removing a feature or process, update or remove affected
  documentation and diagrams in the same change.
