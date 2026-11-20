# Public API (synthetic demo data)

The Acme Notes REST API uses bearer tokens created under Settings > API tokens.

The API is rate limited to 100 requests per minute per token. Requests above the limit
receive HTTP 429 with a Retry-After header.

Webhooks are retried up to 5 times with exponential backoff when the receiver returns an error.
