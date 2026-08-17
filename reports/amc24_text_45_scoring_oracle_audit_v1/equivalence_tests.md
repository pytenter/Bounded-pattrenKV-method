# Equivalence Tests

Targeted tests are in:

```text
tests/test_amc_source_answer_parser.py
tests/test_amc24_text_45_scoring_oracle.py
```

Covered equivalent variants:

- integers with final-answer extraction;
- `\frac{39}{7}`, `\dfrac{39}{7}`, `\tfrac{39}{7}`, and `39/7`;
- `\frac12` to `\frac{1}{2}`;
- `15\sqrt{7}`, `15 \sqrt{7}`, and `15\sqrt7`;
- `(0,\frac{1}{2})`, `(0, \frac12)`, and `(0,1/2)`;
- `[\frac34,\frac78]`, `[3/4,7/8]`, and `[\frac{3}{4}, \frac{7}{8}]`;
- `\frac{\pi}{2}-2\alpha`, `\dfrac{\pi}{2} - 2\alpha`, `\pi/2 - 2\alpha`, and unicode `π/2 - 2α`;
- `-34` and `- 34`;
- `D`, `d`, and boxed/text-wrapped `D`.

Status: `PASS`.
