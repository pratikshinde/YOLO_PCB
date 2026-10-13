import { useSyncExternalStore } from 'react';
import { Directory, File, Paths } from 'expo-file-system';

export type Sensitivity = 'high' | 'normal' | 'low' | 'custom';

/** Version of the saved settings file: 3 = missing-part check with the YOLO part classifier (the old colour-check settings are gone). */
export const SETTINGS_VERSION = 3;

export type Settings = {
  version: number;
  sensitivity: Sensitivity;
  /** a part counts as MISSING when its score p(missing) is at least this (lower = finds more, also more false alarms) */
  missing_threshold: number;
  /** parts from this score up to the missing threshold are listed as "doubtful" (look at them; they do not fail the board) */
  doubt_threshold: number;
  /** 1 = thorough (every part is looked at through 9 windows, median score: more robust, slower), 0 = fast (one window) */
  thorough: number;
  min_inliers: number;
  min_inlier_ratio: number;
  /** the board frame the model was trained on: not meant to be changed */
  board_long_mm: number;
  board_short_mm: number;
  px_per_mm: number;
};

/** High / Normal / Low sensitivity = missing threshold. On the held-out boards every value from 0.2 to 0.5 gave the same result. */
export const PRESETS = {
  high: { missing_threshold: 0.25 },
  normal: { missing_threshold: 0.35 },
  low: { missing_threshold: 0.5 },
} as const;

export const DEFAULTS: Settings = {
  version: SETTINGS_VERSION,
  sensitivity: 'normal',
  missing_threshold: 0.35,
  doubt_threshold: 0.1,
  thorough: 1,
  min_inliers: 30,
  min_inlier_ratio: 0.1,
  board_long_mm: 116,
  board_short_mm: 83,
  px_per_mm: 20,
};

/** Parameters for the native check (same names as scripts/golden_compare.py). */
export function toParams(s: Settings): Record<string, number> {
  return {
    missing_threshold: s.missing_threshold,
    doubt_threshold: Math.min(s.doubt_threshold, s.missing_threshold),
    window_count: s.thorough >= 1 ? 9 : 1,
    min_inliers: Math.round(s.min_inliers),
    min_inlier_ratio: s.min_inlier_ratio,
    max_input_side: 4000,
    ...referenceParams(s),
  };
}

/** The parameters that define the canonical board frame (the reference must be rebuilt when they change). */
export function referenceParams(s: Settings): Record<string, number> {
  return { board_w_mm: s.board_short_mm, board_h_mm: s.board_long_mm, px_per_mm: s.px_per_mm };
}

export const canonicalKey = (s: Settings) => `${s.board_short_mm}x${s.board_long_mm}@${s.px_per_mm}`;

export function sensitivityOf(s: Pick<Settings, 'missing_threshold'>): Sensitivity {
  for (const k of ['high', 'normal', 'low'] as const) if (s.missing_threshold === PRESETS[k].missing_threshold) return k;
  return 'custom';
}

/** Settings of an older app version (colour check, fail policy ...) mean something else now: start from the defaults. */
function migrate(saved: Record<string, unknown>): Settings {
  if (Number(saved.version ?? 1) < SETTINGS_VERSION) return { ...DEFAULTS };
  const out = { ...DEFAULTS, ...saved } as Settings;
  out.sensitivity = sensitivityOf(out);
  return out;
}

// ------------------------------------------------------------------ store

let current: Settings = DEFAULTS;
let loaded = false;
const listeners = new Set<() => void>();
const file = () => new File(new Directory(Paths.document, 'pcb'), 'settings.json');

function emit() {
  listeners.forEach((l) => l());
}

export function getSettings(): Settings {
  return current;
}

export function loadSettings(): Settings {
  if (loaded) return current;
  loaded = true;
  try {
    if (file().exists) current = migrate(JSON.parse(file().textSync()));
  } catch {
    current = DEFAULTS;
  }
  emit();
  return current;
}

export function updateSettings(patch: Partial<Settings>): Settings {
  const next = { ...current, ...patch };
  if ('missing_threshold' in patch) next.sensitivity = sensitivityOf(next);
  current = next;
  try {
    file().create({ intermediates: true, overwrite: true });
    file().write(JSON.stringify(current, null, 1));
  } catch {
    // keep the in-memory value; it is saved again with the next change
  }
  emit();
  return current;
}

export function setSensitivity(level: Exclude<Sensitivity, 'custom'>): Settings {
  return updateSettings({ ...PRESETS[level] });
}

export function resetSettings(): Settings {
  return updateSettings({ ...DEFAULTS });
}

export function useSettings(): Settings {
  loadSettings();
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getSettings,
    getSettings
  );
}
