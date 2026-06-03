# State Representation

The baseline network input is a board tensor from the current player's perspective.

- **Current shape:** `4 x 6 x 6` for the local smoke baseline.
- **Full target shape:** `4 x 15 x 15`.
- **Channel 0:** Current player's stones.
- **Channel 1:** Opponent stones.
- **Channel 2:** Last move location.
- **Channel 3:** Reserved compatibility plane. It is disabled for Gomoku v1 and always encoded as zeros.
- **Legal moves:** Computed from empty intersections at runtime.
- **Action index:** Flatten row-major board coordinates with `action = row * board_size + col`.
- **Action space size:** `width * height`, currently `36` for the smoke baseline and `225` for a 15 x 15 board.

## Notes

- Avoid including Python object state in training samples. Store tensors, policy targets, value targets, and metadata only.
- Keep state encoding deterministic so self-play data can be reproduced.
- Do not reintroduce direct move-count parity or side-to-move shortcuts without a documented experiment and leakage checks.

## Related

- [Rules](Rules.md)
- [Network Architecture](NetworkArchitecture.md)
