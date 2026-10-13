import { Directory, File } from 'expo-file-system';
import { router, Stack, useLocalSearchParams } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { BoardView, type BoxSpec } from '../../components/BoardView';
import { Btn, Card, Chip, H, Muted, Row } from '../../components/ui';
import type { InspectionResult, PartResult } from '../../../modules/golden-comparator';
import { fmtDate, mm1, partLabel, score2, verdictLine } from '../../lib/format';
import { inspectionDir, loadResult } from '../../lib/storage';
import { C, DECISION_COLOR, PART_COLOR } from '../../lib/theme';

const ZOOMS = [1, 2, 3, 4];
const BOARD_MM: [number, number] = [83, 116];

function PartRow({ p, selected, onPress }: { p: PartResult; selected: boolean; onPress: () => void }) {
  const color = p.status === 'missing' ? PART_COLOR.missing : PART_COLOR.doubtful;
  return (
    <Pressable onPress={onPress} style={[s.part, selected && s.partSel]}>
      <View style={[s.dot, { backgroundColor: color }]} />
      <View style={{ flex: 1 }}>
        <Text style={s.pTitle}>
          {p.name}
          <Text style={{ color }}>  {p.status === 'missing' ? 'MISSING' : 'doubtful'}</Text>
        </Text>
        <Muted>
          score {score2(p.p_missing)}  |  at {mm1(p.centre_mm[0])}, {mm1(p.centre_mm[1])} mm from the top-left corner of the board
        </Muted>
      </View>
    </Pressable>
  );
}

export default function ResultScreen() {
  const { id: rawId } = useLocalSearchParams<{ id: string }>();
  const id = String(rawId);
  const { width: winW } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [zoom, setZoom] = useState(1);
  const [showDoubtful, setShowDoubtful] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [allOpen, setAllOpen] = useState(false);
  const scroller = useRef<ScrollView>(null);

  useEffect(() => setResult(loadResult(id)), [id]);

  const dir = useMemo(() => inspectionDir(id), [id]);

  if (!result) {
    return (
      <View style={s.center}>
        <Muted>This inspection could not be found.</Muted>
      </View>
    );
  }
  if (!Array.isArray(result.missing)) {
    return (
      <View style={s.center}>
        <Muted>This result was made by an older version of the app (colour check) and cannot be shown.</Muted>
      </View>
    );
  }

  const contentW = winW - 32;
  const boardW = contentW * zoom;
  const k = boardW / BOARD_MM[0];
  const color = DECISION_COLOR[result.decision];
  const missing = result.missing;
  const doubtful = result.doubtful ?? [];

  const boxes: BoxSpec[] = [];
  const addBox = (p: PartResult, c: string, width: number) => {
    const sel = p.region_id === selected;
    // black edge first (a thicker black border under the coloured one), so the box also stands out on a bright spot
    boxes.push({ key: `k${p.region_id}`, box: boxAround(p), color: '#000000', width: width + 3 });
    boxes.push({ key: p.region_id, box: boxAround(p), color: sel ? '#ffffff' : c, width: sel ? width + 2 : width, label: partLabel(p) });
  };
  if (showDoubtful) doubtful.forEach((p) => addBox(p, PART_COLOR.doubtful, 2));
  missing.forEach((p) => addBox(p, PART_COLOR.missing, 4));

  const alignedFile = new File(dir, 'aligned.jpg');
  const inputFile = new File(dir, result.input ?? 'input.jpg');

  const focusPart = (p: PartResult) => {
    setSelected(p.region_id);
    scroller.current?.scrollTo({ x: Math.max(0, p.centre_mm[0] * k - contentW / 2), animated: true });
  };

  const share = async (f: File, mimeType: string, dialogTitle: string) => {
    try {
      if (!(await Sharing.isAvailableAsync())) return Alert.alert('Sharing is not available on this device');
      await Sharing.shareAsync(f.uri, { mimeType, dialogTitle });
    } catch (e) {
      Alert.alert('Share failed', String((e as Error)?.message ?? e));
    }
  };

  const saveToFolder = async () => {
    try {
      const target = await Directory.pickDirectoryAsync();
      const jpg = target.createFile(`pcb_${id}_annotated.jpg`, 'image/jpeg');
      jpg.write(await new File(dir, 'annotated.jpg').bytes());
      const json = target.createFile(`pcb_${id}_result.json`, 'application/json');
      json.write(JSON.stringify(result, null, 1));
      Alert.alert('Saved', 'annotated JPG and result JSON were saved to the chosen folder.');
    } catch (e) {
      const msg = String((e as Error)?.message ?? e);
      if (!/cancel/i.test(msg)) Alert.alert('Save failed', msg);
    }
  };

  const a = result.alignment;
  return (
    <ScrollView contentContainerStyle={[s.page, { paddingBottom: 24 + insets.bottom }]}>
      <Stack.Screen options={{ title: `Result: ${result.decision}` }} />

      <View style={[s.banner, { backgroundColor: color }]}>
        <Text style={s.bannerBig}>{result.decision}</Text>
        <Text style={s.bannerSub}>{verdictLine(result)}</Text>
      </View>

      {result.decision === 'RETAKE' ? (
        <Card>
          <H>Photo not usable</H>
          <Muted>{result.reason}</Muted>
          <Muted>Retake the photo: the whole board inside the frame with a little background around it, flat, sharp, no strong reflections.</Muted>
          {inputFile.exists ? <Image source={{ uri: inputFile.uri }} style={s.input} resizeMode="contain" resizeMethod="resize" /> : null}
          <Btn title="Retake photo" kind="primary" onPress={() => router.replace('/inspect')} />
        </Card>
      ) : (
        <>
          <Row style={{ flexWrap: 'wrap' }}>
            <Muted>Zoom</Muted>
            {ZOOMS.map((z) => (
              <Chip key={z} label={`${z}x`} active={zoom === z} onPress={() => setZoom(z)} />
            ))}
            {doubtful.length ? <Chip label={`Doubtful ${doubtful.length}`} color={PART_COLOR.doubtful} active={showDoubtful} onPress={() => setShowDoubtful((v) => !v)} /> : null}
          </Row>

          <ScrollView ref={scroller} horizontal nestedScrollEnabled style={s.board} showsHorizontalScrollIndicator>
            <BoardView uri={alignedFile.uri} boardMm={BOARD_MM} widthPx={boardW} boxes={boxes} />
          </ScrollView>
          <Muted>The board as the app saw it (straightened to the golden reference). Yellow box: component missing. Cyan box: doubtful, have a look.</Muted>

          {missing.length ? (
            <Card>
              <H style={{ fontSize: 15 }}>Missing components</H>
              {missing.map((p) => (
                <PartRow key={p.region_id} p={p} selected={selected === p.region_id} onPress={() => focusPart(p)} />
              ))}
            </Card>
          ) : null}

          {doubtful.length ? (
            <Card>
              <H style={{ fontSize: 15 }}>Doubtful ({doubtful.length}): not counted, but look at them</H>
              <Muted>The model is not sure about these parts. They do not change the result.</Muted>
              {doubtful.map((p) => (
                <PartRow key={p.region_id} p={p} selected={selected === p.region_id} onPress={() => { setShowDoubtful(true); focusPart(p); }} />
              ))}
            </Card>
          ) : null}

          <Card>
            <H style={{ fontSize: 15 }}>Details</H>
            <Muted>
              {fmtDate(result.created ?? id)} | source {result.source} | {result.time_s}s
              {result.timing ? ` (align ${result.timing.prepare_s}s, local alignment ${result.timing.local_alignment_s}s, ${result.timing.crops} part windows ${result.timing.parts_s}s)` : ''}
            </Muted>
            {a ? (
              <Muted>
                Alignment: {a.method}, {a.inliers} inliers ({Math.round(a.inlier_ratio * 100)}%), residual {a.residual_mm.toFixed(2)} mm, rotation{' '}
                {a.rotation_deg.toFixed(1)} deg{a.margin_added ? ', board touched the photo edge (margin added)' : ''}
              </Muted>
            ) : null}
            {result.local_alignment ? (
              <Muted>
                Local alignment moved {result.local_alignment.regions_shifted} part windows (largest {result.local_alignment.max_shift_mm} mm)
                {result.exposure_gain != null ? `; exposure gain ${result.exposure_gain}` : ''}
              </Muted>
            ) : null}
            <Muted>
              {result.counts.parts} parts checked | missing at score {result.model?.threshold ?? '?'} or more, doubtful from {result.model?.doubt ?? '?'} | model {result.model?.name ?? '?'}
            </Muted>
            <Pressable onPress={() => setAllOpen((v) => !v)}>
              <Text style={s.link}>{allOpen ? 'Hide' : 'Show'} the score of every part {allOpen ? '▾' : '▸'}</Text>
            </Pressable>
            {allOpen ? <Muted>{result.parts.map((p) => `${p.name} ${score2(p.p_missing)}`).join('   ')}</Muted> : null}
          </Card>

          <Row style={{ flexWrap: 'wrap' }}>
            <Btn small title="Share image" onPress={() => share(new File(dir, 'annotated.jpg'), 'image/jpeg', 'Annotated inspection image')} />
            <Btn small title="Share JSON" onPress={() => share(new File(dir, 'result.json'), 'application/json', 'Inspection result')} />
            <Btn small title="Save to folder..." onPress={saveToFolder} />
          </Row>
          <Btn title="Inspect another board" kind="primary" onPress={() => router.replace('/inspect')} />
        </>
      )}
    </ScrollView>
  );
}

/** The part box grown by 1 mm and moved by the local alignment shift (where the model looked). */
function boxAround(p: PartResult): [number, number, number, number] {
  const [x, y, w, h] = p.bbox_mm;
  return [x + p.shift_mm[0] - 1, y + p.shift_mm[1] - 1, w + 2, h + 2];
}

const s = StyleSheet.create({
  page: { padding: 16, gap: 12, paddingBottom: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  banner: { borderRadius: 16, paddingVertical: 22, alignItems: 'center' },
  bannerBig: { color: '#fff', fontSize: 56, fontWeight: '900', letterSpacing: 2 },
  bannerSub: { color: 'rgba(255,255,255,0.95)', fontSize: 16, fontWeight: '600', marginTop: 2, paddingHorizontal: 16, textAlign: 'center' },
  board: { borderRadius: 8, backgroundColor: '#000', flexGrow: 0 },
  input: { width: '100%', height: 260, backgroundColor: '#000', borderRadius: 8 },
  part: { flexDirection: 'row', gap: 10, paddingVertical: 8, paddingHorizontal: 6, borderRadius: 8, alignItems: 'center' },
  partSel: { backgroundColor: C.card2 },
  dot: { width: 12, height: 12, borderRadius: 6 },
  pTitle: { color: C.text, fontWeight: '700', fontSize: 15 },
  link: { color: C.accent, fontSize: 13, marginTop: 4 },
});
