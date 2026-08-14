# Request Isolation

Unit tests mutate request B score tails and verify request A probabilities are unchanged; the production mask is row-local and broadcast across GQA heads.
