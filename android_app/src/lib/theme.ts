export const C = {
  bg: '#0e1116',
  card: '#171c24',
  card2: '#1f2631',
  border: '#2b3441',
  text: '#e8edf3',
  muted: '#8b97a8',
  accent: '#3b82f6',
  green: '#22c55e',
  red: '#ef4444',
  amber: '#f59e0b',
  white: '#ffffff',
} as const;

/** Colours of the parts on the board image (same hues as the annotated.jpg written by the native module). */
export const PART_COLOR = {
  missing: '#ffe600', // yellow: a red box is not visible on the red board
  doubtful: '#00e5ff', // cyan
} as const;

export const DECISION_COLOR = { PASS: C.green, FAIL: C.red, RETAKE: C.amber } as const;
