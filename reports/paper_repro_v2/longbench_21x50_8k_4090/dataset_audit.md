# Dataset Audit

Experiment: PatternKV paper-v2 configuration-aligned LongBench reproduction with an 8K input cap.
Run scope: 21 tasks x 50 samples per task x 3 methods.

| task | local path | local samples | selected samples | max_gen | metric | prompt_template_hash |
| --- | --- | ---: | ---: | ---: | --- | --- |
| narrativeqa | `/root/Block-kvcache-experiment/data/LongBench/data/narrativeqa.jsonl` | 200 | 50 | 128 | qa_f1 | `aada7ed24b06abace5b0045ac3bcfe06fc831a6cf434f0ecfd99fb7b3d299e1c` |
| qasper | `/root/Block-kvcache-experiment/data/LongBench/data/qasper.jsonl` | 200 | 50 | 128 | qa_f1 | `0fbdd123fe7f83d6d6a9c583ca28cb29d68e523bf8fec01a0cf2dcd11037775d` |
| multifieldqa_en | `/root/Block-kvcache-experiment/data/LongBench/data/multifieldqa_en.jsonl` | 150 | 50 | 64 | qa_f1 | `20b4666a2de8a1f701bdb6c4e015fa9f8758d64b361d841f7766df6c9ceba770` |
| multifieldqa_zh | `/root/Block-kvcache-experiment/data/LongBench/data/multifieldqa_zh.jsonl` | 200 | 50 | 64 | qa_f1_zh | `344b3a4e557cd5470c661b3d6b374339717e1257a6c01f628d22322d4a554b94` |
| hotpotqa | `/root/Block-kvcache-experiment/data/LongBench/data/hotpotqa.jsonl` | 200 | 50 | 32 | qa_f1 | `9ec4ae308865bd0c62af20b3dc7b12b31f2b447f55684bd88c1aa40f2f636deb` |
| 2wikimqa | `/root/Block-kvcache-experiment/data/LongBench/data/2wikimqa.jsonl` | 200 | 50 | 32 | qa_f1 | `9ec4ae308865bd0c62af20b3dc7b12b31f2b447f55684bd88c1aa40f2f636deb` |
| musique | `/root/Block-kvcache-experiment/data/LongBench/data/musique.jsonl` | 200 | 50 | 32 | qa_f1 | `9ec4ae308865bd0c62af20b3dc7b12b31f2b447f55684bd88c1aa40f2f636deb` |
| dureader | `/root/Block-kvcache-experiment/data/LongBench/data/dureader.jsonl` | 200 | 50 | 128 | rouge_l_zh | `ae53329f51b17908e08b8c6bf366987282b8b38c0ee35275ebd7b79e9b30eff2` |
| gov_report | `/root/Block-kvcache-experiment/data/LongBench/data/gov_report.jsonl` | 200 | 50 | 512 | rouge_l | `91c25d585b189ced91c9280efcb4ecb5dcff9548cd4e3bfae66c68025b569e95` |
| qmsum | `/root/Block-kvcache-experiment/data/LongBench/data/qmsum.jsonl` | 200 | 50 | 512 | rouge_l | `dca23b678fe0183bbd9555b57947022aed1b85739da5ad7e1b4708dd0c71c409` |
| multi_news | `/root/Block-kvcache-experiment/data/LongBench/data/multi_news.jsonl` | 200 | 50 | 512 | rouge_l | `2af3f28c40b902e6535d4910510d51653613c49de9b2fc287c5958395061101d` |
| vcsum | `/root/Block-kvcache-experiment/data/LongBench/data/vcsum.jsonl` | 200 | 50 | 512 | rouge_l_zh | `3e5d8241bc9b512eb4929c1ba80882a44f50b446bc2b1d9d79eff030dd10dc27` |
| trec | `/root/Block-kvcache-experiment/data/LongBench/data/trec.jsonl` | 200 | 50 | 64 | classification | `0eff8bebdba99d91e9d7d91b6ab39455ac74648bea92c9255c9f68c62175b2db` |
| triviaqa | `/root/Block-kvcache-experiment/data/LongBench/data/triviaqa.jsonl` | 200 | 50 | 32 | qa_f1 | `00a802e6377e72fca39b169cf6c28893bedf23ba7286b03ecd15867c3e27a7c4` |
| samsum | `/root/Block-kvcache-experiment/data/LongBench/data/samsum.jsonl` | 200 | 50 | 128 | rouge_l | `a68d884234907eefcaa0eacf7b2997f60fa46603988dceae6a44f9b49f045b3f` |
| lsht | `/root/Block-kvcache-experiment/data/LongBench/data/lsht.jsonl` | 200 | 50 | 64 | classification | `f3edabdf7fae53d6cadfa7a10f3574d114302161c46dacecd5d30763204bbbf3` |
| passage_count | `/root/Block-kvcache-experiment/data/LongBench/data/passage_count.jsonl` | 200 | 50 | 32 | count | `986550b6c61009c629a4535766d94b7b51e2b2b9e1ed169e4ecb54417466e964` |
| passage_retrieval_en | `/root/Block-kvcache-experiment/data/LongBench/data/passage_retrieval_en.jsonl` | 200 | 50 | 32 | retrieval | `39b96b7fa52bdd2aa2bde34ea199158505201b9a0a2c3ce94e80249293f60431` |
| passage_retrieval_zh | `/root/Block-kvcache-experiment/data/LongBench/data/passage_retrieval_zh.jsonl` | 200 | 50 | 32 | retrieval_zh | `c1e2346e245eb365bc33a1503cde2e473948a2aad51c3e4c72893e61ab074d60` |
| lcc | `/root/Block-kvcache-experiment/data/LongBench/data/lcc.jsonl` | 500 | 50 | 64 | code_sim | `6a1f24cf0bb83cd90f1fca8f0dd24801626c2aaea29db834ddc83b950e71190e` |
| repobench-p | `/root/Block-kvcache-experiment/data/LongBench/data/repobench-p.jsonl` | 500 | 50 | 64 | code_sim | `ee9ec182ea5ffeb70a7140d55baea97264a8612d88226d323afa90859c36240c` |
