# Length Representation

The production cache now carries `request_total_tokens` and request-local packed token vectors. Decode append increments these vectors independently for every resident request.
