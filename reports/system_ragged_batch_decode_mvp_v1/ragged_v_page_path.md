# Ragged V Page Path

The V page metadata has the right concepts, but the current page batch reference/operator metadata construction is equal-length oriented. With frozen `group_size=128` and `page_size=128`, packed prefill pages are full pages in this audit, so page counts differ while last-page valid tokens remain 128.
