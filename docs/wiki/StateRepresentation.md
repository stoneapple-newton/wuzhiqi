# State Representation

The baseline network input is a board tensor from the current player's perspective.

- **Shape:** `C x 15 x 15`.
- **Channel 0:** Current player's stones.
- **Channel 1:** Opponent stones.
- **Optional channel 2:** Constant plane for side-to-move or move parity if experiments show it helps.
- **Legal moves:** Computed from empty intersections at runtime.
- **Action index:** Flatten row-major board coordinates with `action = row * board_size + col`.
- **Action space size:** `board_size * board_size`, currently `225` for a 15 x 15 board.

## Notes

- Avoid including Python object state in training samples. Store tensors, policy targets, value targets, and metadata only.
- Keep state encoding deterministic so self-play data can be reproduced.

## Related

- [Rules](Rules.md)
- [Network Architecture](NetworkArchitecture.md)
