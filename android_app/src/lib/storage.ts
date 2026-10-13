import { Directory, File, Paths } from 'expo-file-system';

import { GoldenComparator, type InspectionResult, type RegionsFile } from '../../modules/golden-comparator';
import bundledRegions from '../../assets/golden/regions.json';
import { referenceParams, type Settings } from './settings';

/**
 * App-private storage (Paths.document/pcb):
 *   references/<name>/reference.jpg  the golden reference photo bundled in the APK (never modified)
 *                     canonical.png  rectified board (1660 x 2320 px at 20 px/mm)
 *                     structure.png  pixels the local alignment of every part is measured on
 *                     regions.json   the 86 part boxes (mm) + empty footprints, names, kinds
 *                     meta.json      board size, px/mm, board colour hue, created time
 *   inspections/<timestamp>/input.jpg   original photo (never modified)
 *                           aligned.jpg, annotated.jpg, result.json (+ summary.json for History)
 *   state.json                       { active: <reference name> }
 */
export const DEFAULT_REFERENCE = 'LG_INDICATOR_SMT_RED';

const root = () => new Directory(Paths.document, 'pcb');
export const referencesDir = () => new Directory(root(), 'references');
export const inspectionsDir = () => new Directory(root(), 'inspections');
export const referenceDir = (name: string) => new Directory(referencesDir(), name);
export const inspectionDir = (id: string) => new Directory(inspectionsDir(), id);

/** Directory URI without the trailing slash (what the native module expects). */
export const dirUri = (d: Directory) => d.uri.replace(/\/+$/, '');

export function ensureDir(d: Directory): Directory {
  if (!d.exists) d.create({ intermediates: true, idempotent: true });
  return d;
}

export function readJson<T>(f: File): T | null {
  try {
    return f.exists ? (JSON.parse(f.textSync()) as T) : null;
  } catch {
    return null;
  }
}

export function writeJson(f: File, data: unknown): void {
  f.create({ intermediates: true, overwrite: true });
  f.write(JSON.stringify(data, null, 1));
}

/** Image URI that changes when the file changes (React Native caches images by URI). */
export function freshUri(f: File): string {
  return `${f.uri}?v=${f.exists ? (f.lastModified ?? f.size) : 0}`;
}

export function extOf(uri: string): string {
  const m = /\.([a-zA-Z0-9]{2,5})(?:\?.*)?$/.exec(uri);
  const e = (m?.[1] ?? 'jpg').toLowerCase();
  return e === 'jpeg' ? 'jpg' : e;
}

// ------------------------------------------------------------------ golden reference (bundled, fixed)

export type ReferenceMeta = {
  name: string;
  created: string;
  source: 'bundled' | 'camera' | 'gallery';
  photo?: string; // file name of the original photo inside the reference folder (default reference.jpg)
  board_mm: [number, number];
  px_per_mm: number;
  hue: number;
  colour: string;
  regions: number;
  note?: string;
};

export function colourOfHue(hue: number): string {
  if (hue <= 12 || hue >= 168) return 'red';
  if (hue >= 35 && hue <= 85) return 'green';
  if (hue >= 90 && hue <= 130) return 'blue';
  return `hue${hue}`;
}

export function getActiveReferenceName(): string {
  return DEFAULT_REFERENCE; // the part classifier belongs to this golden reference only
}

export function referenceReady(name: string): boolean {
  const d = referenceDir(name);
  return new File(d, 'canonical.png').exists && new File(d, 'regions.json').exists && new File(d, 'structure.png').exists;
}

export function loadRegions(name: string): RegionsFile | null {
  return readJson<RegionsFile>(new File(referenceDir(name), 'regions.json'));
}

export function loadMeta(name: string): ReferenceMeta | null {
  return readJson<ReferenceMeta>(new File(referenceDir(name), 'meta.json'));
}

const frameMatches = (m: ReferenceMeta | null, s: Settings) =>
  !!m && m.px_per_mm === s.px_per_mm && m.board_mm[0] === s.board_short_mm && m.board_mm[1] === s.board_long_mm;

/** Installs the bundled golden reference (grf.jpeg + the 86 named part boxes) at the first start, or again when it is incomplete / has another scale. */
export async function ensureDefaultReference(settings: Settings, force = false): Promise<void> {
  const dir = referenceDir(DEFAULT_REFERENCE);
  if (!force && referenceReady(DEFAULT_REFERENCE) && frameMatches(loadMeta(DEFAULT_REFERENCE), settings)) return;
  ensureDir(dir);
  const photo = new File(dir, 'reference.jpg');
  if (photo.exists) photo.delete();
  await GoldenComparator.installBundledGolden(photo.uri); // native copy of the photo packaged in the APK
  const res = await GoldenComparator.prepareReference(
    photo.uri,
    dirUri(dir),
    referenceParams(settings),
    bundledRegions as unknown as RegionsFile
  );
  if (!res.ok) throw new Error(`Bundled golden reference unusable: ${res.retake}`);
  writeJson(new File(dir, 'meta.json'), {
    name: DEFAULT_REFERENCE,
    created: new Date().toISOString(),
    source: 'bundled',
    board_mm: [settings.board_short_mm, settings.board_long_mm],
    px_per_mm: settings.px_per_mm,
    hue: res.hue,
    colour: colourOfHue(res.hue),
    regions: res.regions_count,
    note: 'Bundled golden reference: GOOD board after SMT reflow, LEDs / switch not fitted.',
  } satisfies ReferenceMeta);
}

// ------------------------------------------------------------------ inspections

export type InspectionSummary = {
  id: string;
  created: string;
  decision: InspectionResult['decision'];
  reason: string;
  reference: string;
  source: string;
  /** missing parts (decide FAIL) and doubtful parts (look at them) */
  missing: number;
  doubtful: number;
  parts: number;
  thumb: string;
};

export function newInspectionId(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

export function loadResult(id: string): InspectionResult | null {
  return readJson<InspectionResult>(new File(inspectionDir(id), 'result.json'));
}

export function summaryOf(r: InspectionResult): InspectionSummary {
  return {
    id: r.id ?? '',
    created: r.created ?? '',
    decision: r.decision,
    reason: r.reason,
    reference: r.reference ?? '',
    source: r.source ?? '',
    missing: r.counts?.missing ?? 0,
    doubtful: r.counts?.doubtful ?? 0,
    parts: r.counts?.parts ?? 0,
    thumb: r.decision === 'RETAKE' ? (r.input ?? 'input.jpg') : 'annotated.jpg',
  };
}

export function listInspections(): InspectionSummary[] {
  if (!inspectionsDir().exists) return [];
  const out: InspectionSummary[] = [];
  for (const e of inspectionsDir().list()) {
    if (!(e instanceof Directory)) continue;
    let s = readJson<InspectionSummary>(new File(e, 'summary.json'));
    if (!s || s.missing == null) {
      // no summary, or one from the old colour-check version: rebuild it from result.json (old results show as unreadable details only)
      const r = readJson<InspectionResult>(new File(e, 'result.json'));
      if (!r) continue;
      s = summaryOf({ ...r, id: r.id ?? e.name });
    }
    out.push(s);
  }
  return out.sort((a, b) => b.id.localeCompare(a.id));
}
