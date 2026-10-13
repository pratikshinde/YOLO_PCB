import { File } from 'expo-file-system';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Image, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { GoldenComparator, type ModelInfo } from '../../modules/golden-comparator';
import { Btn, Card, H, Muted, PhotoTips } from '../components/ui';
import { freshUri, getActiveReferenceName, loadRegions, referenceDir } from '../lib/storage';
import { C } from '../lib/theme';

type Info = { parts: number; thumb: string };

export default function Home() {
  const [info, setInfo] = useState<Info | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const insets = useSafeAreaInsets();

  useFocusEffect(
    useCallback(() => {
      const name = getActiveReferenceName();
      const regions = loadRegions(name)?.regions ?? [];
      setInfo({ parts: regions.filter((r) => r.kind === 'part').length, thumb: freshUri(new File(referenceDir(name), 'canonical.png')) });
      GoldenComparator.modelInfo().then(setModel).catch(() => setModel(null));
    }, [])
  );

  const cv = model?.cross_validation;
  const th = model?.cross_validation_thorough;
  return (
    <ScrollView contentContainerStyle={[s.page, { paddingBottom: 24 + insets.bottom }]}>
      <Card>
        <H style={{ fontSize: 13, color: C.muted, fontWeight: '600' }}>MISSING-PART CHECK  -  LG INDICATOR, RED BOARD</H>
        {info ? (
          <View style={s.ref}>
            <Image source={{ uri: info.thumb }} style={s.thumb} resizeMode="cover" />
            <View style={{ flex: 1, gap: 3 }}>
              <Text style={s.big}>
                {info.parts} <Text style={s.bigUnit}>SMT parts checked</Text>
              </Text>
              <Muted>Photograph the whole board; every part is looked at and judged present or missing.</Muted>
              {model ? <Muted>Model {model.name} ({model.architecture}), trained {model.trained}</Muted> : null}
            </View>
          </View>
        ) : null}
        {th ? (
          <Muted>
            Test on whole boards the model had never seen: {th.missed} missing parts missed, {th.false_calls} good parts called missing.
          </Muted>
        ) : cv ? (
          <Muted>
            Test on boards the model had never seen: {cv.missing_crops - cv.missed_at_threshold} of {cv.missing_crops} missing parts found, {cv.false_calls_at_threshold} false alarm in{' '}
            {cv.present_crops} good parts.
          </Muted>
        ) : null}
        {model?.not_proven_part_types?.length ? (
          <Text style={s.warn}>
            Proven on resistors only. Not yet proven (few or no real missing examples): {model.not_proven_part_types.join(', ')}.
          </Text>
        ) : null}
      </Card>

      <Btn title="Inspect new board" kind="primary" onPress={() => router.push('/inspect')} style={s.main} />
      <View style={s.grid}>
        <Btn title="History" onPress={() => router.push('/history')} style={s.half} />
        <Btn title="Settings" onPress={() => router.push('/settings')} style={s.half} />
      </View>

      <PhotoTips />
    </ScrollView>
  );
}

const s = StyleSheet.create({
  page: { padding: 16, gap: 12 },
  ref: { flexDirection: 'row', gap: 14, alignItems: 'center' },
  thumb: { width: 96, height: 134, borderRadius: 8, backgroundColor: '#000' },
  big: { color: C.text, fontSize: 26, fontWeight: '800', marginTop: 4 },
  bigUnit: { fontSize: 14, fontWeight: '500', color: C.muted },
  warn: { color: C.amber, fontSize: 12, lineHeight: 17 },
  main: { minHeight: 64 },
  grid: { flexDirection: 'row', gap: 12 },
  half: { flex: 1 },
});
