# Fused Operator Design

The MVP fused kernel launches once per decode. Grid x spans `B*nh`, grid y spans output channels. Each block reduces over sequence tokens for one `(request, query-head, output-channel)` scalar, loads V2/V4 compressed payload directly, applies affine and centroid correction, and writes the final Value vector without materializing page-local Value tensors.
