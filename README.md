# sd-forge-condelta

Forge Neo always-on extension for using the existing Negative Prompt field as a
ConDelta-style low-CFG negative prompt.

The extension adds a normal extension accordion named `ConDelta low-CFG negative`
with:

- `Activation CFG threshold`, default `1.0`, range `1.0..24.0`, step `0.5`
- `ConDelta strength`, default `0.6`, range `0.0..1.0`, step `0.05`
- `Also use native negative prompt above CFG 1.0`, default off

When the current pass CFG is at or below the threshold and the negative prompt is
not empty, the positive conditioning is changed as:

```text
positive - strength * (negative - blank)
```

At CFG 1.0 this uses ConDelta only. Above CFG 1.0 and at or below the threshold,
the checkbox controls whether Forge Neo's native negative conditioning is also
kept or replaced by blank unconditional conditioning.
