import { NativeModule, requireNativeModule, requireNativeView } from 'expo';
import { forwardRef, useImperativeHandle, useRef } from 'react';
import type { StyleProp, ViewStyle } from 'react-native';

export type Box = [number, number, number, number]; // x, y, w, h in mm (board frame, origin top-left)

export type RegionKind = 'part' | 'empty';

export type Region = {
  id: string;
  name: string;
  kind: RegionKind | '';
  bbox_mm: Box;
  centre_mm?: [number, number];
  part_fraction?: number;
  texture_std?: number;
  body_px?: number;
};

export type RegionsFile = {
  canonical_px: [number, number];
  px_per_mm: number;
  board_mm: [number, number];
  regions: Region[];
};

export type PartStatus = 'missing' | 'doubtful' | 'present';

/** One of the 86 SMT parts: its score p_missing (0 = surely there, 1 = surely missing) and where it was looked for. */
export type PartResult = {
  region_id: string;
  name: string;
  status: PartStatus;
  /** the score that decides: the median over the windows in thorough mode */
  p_missing: number;
  /** score of the centre window alone */
  p_centre?: number;
  bbox_mm: Box;
  centre_mm: [number, number];
  shift_mm: [number, number];
};

export type Decision = 'PASS' | 'FAIL' | 'RETAKE';

export type InspectionResult = {
  decision: Decision;
  reason: string;
  counts: { parts: number; missing: number; doubtful: number };
  /** parts with p_missing >= the missing threshold (they decide FAIL), highest score first */
  missing: PartResult[];
  /** parts between the doubt threshold and the missing threshold: look at them, they do not fail the board */
  doubtful: PartResult[];
  /** all parts, highest score first */
  parts: PartResult[];
  alignment?: {
    method: string;
    inliers: number;
    matches: number;
    inlier_ratio: number;
    residual_mm: number;
    rotation_deg: number;
    coverage: number;
    full_frame_input: boolean;
    sharpness: number;
    margin_added?: boolean;
  };
  /** where the board around the parts lay relative to the golden reference (bent board / lens): local correction of every part window */
  local_alignment?: { regions_shifted: number; max_shift_mm: number };
  exposure_gain?: number;
  /** windows per part: 9 = thorough (median score), 1 = fast */
  windows?: number;
  timing?: { prepare_s: number; local_alignment_s: number; parts_s: number; crops: number };
  model?: { name: string; threshold: number; doubt: number };
  time_s: number;
  params?: Record<string, number>;
  // added by the app
  id?: string;
  created?: string;
  reference?: string;
  source?: 'camera' | 'gallery';
  input?: string;
};

export type ModelInfo = {
  name: string;
  architecture: string;
  trained: string;
  threshold: number;
  doubt: number;
  exposure_norm: boolean;
  proven_part_types: string[];
  not_proven_part_types: string[];
  note?: string;
  cross_validation?: {
    folds: number;
    missing_crops: number;
    present_crops: number;
    missed_at_threshold: number;
    false_calls_at_threshold: number;
    good_photos_failing: string;
    removed_photos_exact: string;
  };
  /** the same held-out boards judged in thorough mode (median over 9 windows per part) */
  cross_validation_thorough?: {
    rule: string;
    threshold: number;
    missed: string;
    false_calls: string;
    good_photos_failing: string;
    removed_photos_exact: string;
    lowest_missing_score: number;
    highest_present_score: number;
  };
};

export type PrepareResult =
  | {
      ok: true;
      regions_count: number;
      hue: number;
      full_frame: boolean;
      sharpness: number;
      outline: number[]; // 8 numbers: TL, TR, BR, BL in photo px
      image_size: [number, number];
      regions: RegionsFile;
    }
  | { ok: false; retake: string };

type NativeEvents = { onProgress: (e: { stage: string }) => void };

declare class GoldenComparatorNative extends NativeModule<NativeEvents> {
  defaultParams(): string;
  installBundledGolden(destUri: string): Promise<string>;
  prepareReference(imageUri: string, regionsJson: string | null, paramsJson: string | null, outDir: string): Promise<string>;
  modelInfo(): Promise<string>;
  inspect(referenceDir: string, imageUri: string, outDir: string, paramsJson: string | null): Promise<string>;
}

const native = requireNativeModule<GoldenComparatorNative>('GoldenComparator');

const p = (params: object | null | undefined) => (params ? JSON.stringify(params) : null);

export const GoldenComparator = {
  defaultParams: (): Record<string, number> => JSON.parse(native.defaultParams()),

  /** Copies the golden reference photo bundled in the APK to destUri (a file:// URI inside the app's storage). */
  installBundledGolden: (destUri: string) => native.installBundledGolden(destUri),

  /** Rectifies the golden photo and writes canonical.png / structure.png / regions.json / outline.jpg into outDir (done once, at the first start). */
  async prepareReference(
    imageUri: string,
    outDir: string,
    params?: object,
    regions?: RegionsFile | null
  ): Promise<PrepareResult> {
    return JSON.parse(await native.prepareReference(imageUri, regions ? JSON.stringify(regions) : null, p(params), outDir));
  },

  /** The model card shipped in the APK (name, thresholds, cross-validation numbers, which part types are proven). */
  async modelInfo(): Promise<ModelInfo> {
    return JSON.parse(await native.modelInfo());
  },

  /** The missing-part check. Writes result.json, aligned.jpg and annotated.jpg into outDir. */
  async inspect(
    referenceDir: string,
    imageUri: string,
    outDir: string,
    params?: object,
    onProgress?: (stage: string) => void
  ): Promise<InspectionResult> {
    const sub = onProgress ? native.addListener('onProgress', (e) => onProgress(e.stage)) : null;
    try {
      return JSON.parse(await native.inspect(referenceDir, imageUri, outDir, p(params)));
    } finally {
      sub?.remove();
    }
  },
};

// ------------------------------------------------------------------ camera view

export type GoldenCameraHandle = {
  takePicture(path: string): Promise<{ uri: string }>;
  focusAt(xDp: number, yDp: number): Promise<void>;
};

type CameraProps = {
  style?: StyleProp<ViewStyle>;
  active: boolean;
  onCameraReady?: () => void;
  onCameraError?: (e: { nativeEvent: { message: string } }) => void;
};

const NativeCameraView = requireNativeView('GoldenComparator');

export const GoldenCameraView = forwardRef<GoldenCameraHandle, CameraProps>(function GoldenCameraView(props, ref) {
  const inner = useRef<any>(null);
  useImperativeHandle(ref, () => ({
    takePicture: (path: string) => inner.current.takePicture(path),
    focusAt: (x: number, y: number) => inner.current.focusAt(x, y),
  }));
  return <NativeCameraView ref={inner} {...props} />;
});
