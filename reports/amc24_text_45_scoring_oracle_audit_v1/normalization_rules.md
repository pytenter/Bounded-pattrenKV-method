# Normalization Rules

Normalizer: `normalize_answer` in `evaluation/amc_source_answer_parser.py`.

Version: `amc24_text_normalizer_v1`.

## RULE 1: Strip Outer Whitespace And Final Punctuation

Removes leading/trailing whitespace and terminal period/full stop.

Motivated by all source answer forms and final-answer line extraction.

Does not remove mathematical signs or internal punctuation.

## RULE 2: Strip Final-Answer Math Wrappers

Removes outer `$...$`, `\(...\)`, `\[...\]`, and a single whole-string `\boxed{...}` wrapper.

Motivated by integer, fraction, radical, tuple, interval, expression, and letter final answers.

Does not scan arbitrary reasoning for math expressions.

## RULE 3: Normalize Common LaTeX Fraction Commands

Maps `\dfrac` and `\tfrac` to `\frac`.

Motivated by source answers including `\frac{39}{7}`, `\frac{91}{180}`, and interval endpoints.

Does not simplify rational values or compare decimal approximations.

## RULE 4: Normalize Simple Fraction Shorthand

Maps unambiguous `\frac12` style syntax to `\frac{1}{2}`.

Motivated by tuple and interval variants such as `(0,\frac12)` and `[\frac34,\frac78]`.

Does not parse arbitrary nested TeX.

## RULE 5: Normalize Simple Slash Fractions

Maps bounded forms like `39/7`, `3/4`, `7/8`, `\pi/2`, and `\alpha/2` to `\frac{...}{...}`.

Motivated by source fractions, tuple endpoints, interval endpoints, and `\frac{\pi}{2}-2\alpha`.

Does not evaluate decimal approximations or simplify expressions.

## RULE 6: Normalize Simple Radical Shorthand

Maps `\sqrt7` to `\sqrt{7}` and removes insignificant whitespace around radicals.

Motivated by `15\sqrt{7}`, `\log_2 \frac{7}{\sqrt{3}}`, and `\frac{1 + \sqrt{2}}{2}`.

Does not convert radicals to floating point.

## RULE 7: Remove Harmless LaTeX Sizing/Text Wrappers

Removes `\left`, `\right`, and unwraps simple `\text{...}`.

Motivated by tuple formatting and boxed letter answers such as `\boxed{\text{D}}`.

Does not remove mathematical operators.

## RULE 8: Normalize Whitespace

Removes insignificant whitespace inside answer strings.

Motivated by all expression categories, especially tuple, interval, radical, and symbolic answers.

Does not reorder tuple, interval, or expression components.

## RULE 9: Normalize Unicode Constants

Maps `π` to `\pi`, `α` to `\alpha`, and Unicode minus to `-`.

Motivated by symbolic expression variants.

Does not approximate `\pi` numerically.

## RULE 10: Normalize Single-Letter Answers

Uppercases a single ASCII letter answer.

Motivated by source answer `D` for `12B_04`.

Does not uppercase symbolic commands or multi-character expressions.

## Explicit Non-Rules

- No decimal tolerance.
- No general symbolic simplification.
- No rational reduction as a correctness oracle.
- No LLM judging.
- No gold-aware candidate selection.
- No interval bracket conversion.
- No tuple component reordering.
