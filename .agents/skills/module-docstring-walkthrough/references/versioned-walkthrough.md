# Versioned Matrix Walkthrough Template

Use this template when a versioned module has a documented sibling and works
with matrices or projections. Keep the example small, fixed, and checked.

```text
"""<Plain-language purpose.>

The module accepts <input shape> and returns <output shape>.

Example walkthrough:
    Use this illustrative input. <Explain any row labels.>

        inputs
        <two small input rows>

    Use these fixed, distinct parameters for this example. The model learns
    its real parameters at runtime.

        <query parameter>
        <key parameter>
        <value parameter>

    1. <Project or create the first intermediate values.>

        <queries, keys, and values with calculated numbers>

    2. <Calculate and scale scores.>

        <scores and scaled scores with calculated numbers>

    3. <Normalize or make the next intermediate value.>

        <weights with calculated numbers>

    4. <Create the final output.>

        <output with calculated numbers>

Difference from ``<SiblingClass>``:
    <One or two plain-language sentences about the version difference.>
"""
```

Use the sibling's terms and step order when the calculation is shared. For a
different projection API, show its actual parameter orientation and calculate
the displayed values with that API.
