# Gomoku Rules

- **Current code default:** Local smoke runs use a 6 x 6 board.
- **Full target:** Standard Gomoku uses 15 x 15 intersections.
- **Players:** Black and White alternate moves.
- **First move:** Black plays first.
- **Move:** A player places one stone on an empty intersection.
- **Current code win length:** 4 in a row for the 6 x 6 smoke baseline.
- **Full target win length:** A player wins immediately after creating five or more contiguous stones in a row horizontally, vertically, or diagonally.
- **Draw:** If the board fills with no winning line, the game is a draw.
- **Variant baseline:** Start with free-style Gomoku. Renju restrictions such as forbidden double-threes can be added later only if explicitly selected.

## Related

- [State Representation](StateRepresentation.md)
- [Evaluation Metrics](EvaluationMetrics.md)
