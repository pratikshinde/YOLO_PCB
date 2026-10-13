import type { InspectionResult, PartResult } from '../../modules/golden-comparator';

export const mm1 = (v: number) => (Math.round(v * 10) / 10).toFixed(1);
export const score2 = (v: number) => v.toFixed(2);

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** What decided the result, in words: "2 missing components: R12, Q4", or what was checked for a PASS. */
export function verdictLine(r: InspectionResult): string {
  const missing = r.missing ?? [];
  if (r.decision === 'FAIL') {
    const names = missing.slice(0, 6).map((m) => m.name).join(', ');
    return `${plural(missing.length, 'missing component', 'missing components')}: ${names}${missing.length > 6 ? ' ...' : ''}`;
  }
  if (r.decision !== 'PASS') return r.reason;
  const lines = [`No missing components  (${r.counts?.parts ?? 86} parts checked)`];
  const d = r.counts?.doubtful ?? 0;
  if (d > 0) lines.push(`${plural(d, 'part', 'parts')} doubtful: have a look`);
  return lines.join('\n');
}

/** "R12  missing  0.98" (same wording as the annotated image). */
export function partLabel(p: PartResult): string {
  return `${p.name} ${score2(p.p_missing)}`;
}
