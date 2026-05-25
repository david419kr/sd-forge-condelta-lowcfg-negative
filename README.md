# sd-forge-condelta

Forge Neo always-on extension for ConDelta-style negative prompting.

Settings include `ConDelta prompt mode`:

- `Seamless` keeps the existing Negative Prompt field workflow.
- `Dedicated Prompt` adds a separate one-line `ConDelta negative prompt` field
  below the native Negative Prompt field.

In `Seamless` mode, the extension adds a normal extension accordion named
`ConDelta low-CFG negative` with:

- `Activation CFG threshold`, default `1.0`, range `1.0..24.0`, step `0.5`
- `ConDelta strength`, default `0.6`, range `0.0..2.0`, step `0.05`
  - backend values are accepted from `-100.0..100.0`; edit `ui-config.json`
    if you want the UI control itself to expose a wider range
- `Also use native negative prompt above CFG 1.0`, default off

In `Dedicated Prompt` mode, the accordion only shows `ConDelta strength`. The
dedicated prompt field controls whether ConDelta is applied.

When active, the positive conditioning is changed as:

```text
positive - strength * (negative - blank)
```

In `Seamless` mode, CFG 1.0 uses ConDelta only. Above CFG 1.0 and at or below
the threshold, the checkbox controls whether Forge Neo's native negative
conditioning is also kept or replaced by blank unconditional conditioning.

In `Dedicated Prompt` mode, Forge Neo's native Negative Prompt behavior is left
unchanged and the dedicated ConDelta prompt is applied in addition when it is not
empty.
