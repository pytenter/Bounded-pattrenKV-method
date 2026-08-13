# K vs V Layout Sensitivity

| Property | K/QK | V/AV |
| --- | --- | --- |
| Compute pattern | Q dot K per token, loop over head_dim for each token-pack | Attention weights reduce across tokens into output channels |
| Token traversal | Output dimension is token; QK writes one score per historical token | Token dimension is reduction axis; output dimension is channel |
| Channel traversal | Warp lanes traverse head_dim channels for a fixed packed token | Warp lanes/histogram path tolerate token-strided metadata and payload access better |
| Packed layout | Tight reader expects `[packed_token, channel]` so channel tile is contiguous | V reader already uses `[token, output_pack]`; capacity stride preserves token-major logical access |
| Reuse pattern | Query tile is reused while reading K residual/scales/zeros per token-pack | Attention scalar and centroid histogram dominate; V payload access is less coupled to a transposed channel-contiguous layout |
| Stride sensitivity | High: capacity layout changes fixed-token channel tile from contiguous to spaced by `cap_packs` | Low: S5A-1 measured around 4.99% V2 overhead |
| Static instruction delta | K IMAD/add count rises strongly | V2 IMAD/add count changes modestly |
| Observed overhead | 32K K overhead 33.68%; 16K/24K around 46% | Prior V2 strided overhead around 4.99%; S5A-2 mixed-V/E2E still faster |

Interpretation: K/QK has strong layout-kernel coupling to the tight transposed K layout. V/AV is more compatible with capacity-backed token-strided views.
