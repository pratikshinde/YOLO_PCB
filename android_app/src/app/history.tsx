import { File } from 'expo-file-system';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Muted } from '../components/ui';
import { fmtDate } from '../lib/format';
import { inspectionDir, listInspections, type InspectionSummary } from '../lib/storage';
import { C, DECISION_COLOR, PART_COLOR } from '../lib/theme';

export default function History() {
  const [items, setItems] = useState<InspectionSummary[]>([]);
  const insets = useSafeAreaInsets();
  useFocusEffect(useCallback(() => setItems(listInspections()), []));

  return (
    <FlatList
      data={items}
      keyExtractor={(i) => i.id}
      contentContainerStyle={{ padding: 16, gap: 10, paddingBottom: 24 + insets.bottom }}
      ListEmptyComponent={<Muted style={{ textAlign: 'center', marginTop: 40 }}>No inspections yet.</Muted>}
      renderItem={({ item }) => (
        <Pressable onPress={() => router.push({ pathname: '/result/[id]', params: { id: item.id } })} style={s.row}>
          <Image
            source={{ uri: new File(inspectionDir(item.id), item.thumb).uri }}
            style={s.thumb}
            resizeMode="cover"
            resizeMethod="resize"
          />
          <View style={{ flex: 1, gap: 4 }}>
            <Text style={[s.decision, { color: DECISION_COLOR[item.decision] }]}>{item.decision}</Text>
            <Muted>{item.created ? fmtDate(item.created) : item.id}</Muted>
            {item.decision === 'FAIL' ? (
              <Text style={[s.count, { color: PART_COLOR.missing }]}>
                {item.missing} missing component{item.missing === 1 ? '' : 's'}
              </Text>
            ) : item.decision === 'RETAKE' ? (
              <Muted numberOfLines={2}>{item.reason}</Muted>
            ) : (
              <Muted>
                No missing components{item.doubtful ? `  (${item.doubtful} doubtful)` : ''}
              </Muted>
            )}
          </View>
        </Pressable>
      )}
    />
  );
}

const s = StyleSheet.create({
  row: { flexDirection: 'row', gap: 12, backgroundColor: C.card, borderRadius: 12, borderWidth: 1, borderColor: C.border, padding: 10 },
  thumb: { width: 72, height: 100, borderRadius: 6, backgroundColor: '#000' },
  decision: { fontSize: 20, fontWeight: '800' },
  count: { fontSize: 13, fontWeight: '700' },
});
