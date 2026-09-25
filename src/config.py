"""Project-wide settings shared by training-data and model modules."""

# Each training example and its positional-embedding table use this many
# token positions. Keep the default shared so they cannot silently diverge.
MAX_SEQUENCE_LENGTH = 256
