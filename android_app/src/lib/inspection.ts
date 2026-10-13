import { File } from 'expo-file-system';

import { GoldenComparator, type InspectionResult } from '../../modules/golden-comparator';
import { getSettings, toParams } from './settings';
import {
  dirUri,
  ensureDefaultReference,
  ensureDir,
  extOf,
  getActiveReferenceName,
  inspectionDir,
  newInspectionId,
  referenceDir,
  summaryOf,
  writeJson,
} from './storage';

/**
 * Checks a photo (camera / gallery) for missing components.
 * The original photo is copied as input.<ext> and never modified; the native module writes
 * aligned.jpg / annotated.jpg / result.json next to it. Returns the inspection id.
 */
export async function runInspection(
  photoUri: string,
  source: 'camera' | 'gallery',
  onStage?: (stage: string) => void
): Promise<string> {
  const settings = getSettings();
  const refName = getActiveReferenceName();
  onStage?.('Preparing the golden reference');
  await ensureDefaultReference(settings); // rebuilds only when it is missing or made for another board scale

  const id = newInspectionId();
  const dir = ensureDir(inspectionDir(id));
  const input = new File(dir, `input.${extOf(photoUri)}`);
  new File(photoUri).copySync(input); // the original, never modified

  const res = await GoldenComparator.inspect(dirUri(referenceDir(refName)), input.uri, dirUri(dir), toParams(settings), onStage);
  const full: InspectionResult = {
    ...res,
    id,
    created: new Date().toISOString(),
    reference: refName,
    source,
    input: input.name,
  };
  writeJson(new File(dir, 'result.json'), full);
  writeJson(new File(dir, 'summary.json'), summaryOf(full));
  return id;
}
