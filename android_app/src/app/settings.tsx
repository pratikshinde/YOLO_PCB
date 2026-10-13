import { useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { GoldenComparator, type ModelInfo } from '../../modules/golden-comparator';
import { Btn, Card, H, Muted, Segmented } from '../components/ui';
import { DEFAULTS, resetSettings, setSensitivity, updateSettings, useSettings } from '../lib/settings';
import { C } from '../lib/theme';

function Field({
  label,
  hint,
  value,
  min,
  max,
  onCommit,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const v = Number(text.replace(',', '.'));
    if (!Number.isFinite(v) || v < min || v > max) {
      setText(String(value));
      Alert.alert('Out of range', `${label}: enter a number from ${min} to ${max}.`);
      return;
    }
    if (v !== value) onCommit(v);
  };
  return (
    <View style={s.field}>
      <View style={{ flex: 1 }}>
        <Text style={s.label}>{label}</Text>
        {hint ? <Muted>{hint}</Muted> : null}
      </View>
      <TextInput value={text} onChangeText={setText} onEndEditing={commit} onSubmitEditing={commit} keyboardType="decimal-pad" selectTextOnFocus style={s.input} />
    </View>
  );
}

export default function SettingsScreen() {
  const st = useSettings();
  const insets = useSafeAreaInsets();
  const [model, setModel] = useState<ModelInfo | null>(null);
  useEffect(() => {
    GoldenComparator.modelInfo().then(setModel).catch(() => setModel(null));
  }, []);

  return (
    <ScrollView contentContainerStyle={[s.page, { paddingBottom: 24 + insets.bottom }]} keyboardShouldPersistTaps="handled">
      <Card>
        <H>Sensitivity</H>
        <Segmented
          options={[
            { value: 'high', label: 'High' },
            { value: 'normal', label: 'Normal' },
            { value: 'low', label: 'Low' },
          ]}
          value={st.sensitivity}
          onChange={(v) => setSensitivity(v as 'high' | 'normal' | 'low')}
        />
        <Muted>
          A board FAILS when a component is missing. Every part gets a score between 0 (surely there) and 1 (surely missing); it counts as missing from the
          threshold below. High = 0.25, Normal = 0.35, Low = 0.50.{st.sensitivity === 'custom' ? ' Currently a custom value.' : ''} On the boards tested the
          result was the same for every value from 0.2 to 0.5; a lower threshold is the safer choice for part types that have not been proven.
        </Muted>
      </Card>

      <Card>
        <H>Check mode</H>
        <Segmented
          options={[
            { value: 'thorough', label: 'Thorough' },
            { value: 'fast', label: 'Fast' },
          ]}
          value={st.thorough >= 1 ? 'thorough' : 'fast'}
          onChange={(v) => updateSettings({ thorough: v === 'thorough' ? 1 : 0 })}
        />
        <Muted>
          Thorough: every part is looked at through 9 slightly shifted windows and the median score counts. It takes about twice as long but is more robust against small
          alignment errors (on the boards tested it found all removed parts and gave the fewest false alarms). Fast: one window per part.
        </Muted>
      </Card>

      <Card>
        <H>Thresholds</H>
        <Field label="Missing from score" hint="a part with this score or more is MISSING" value={st.missing_threshold} min={0.05} max={0.95} onCommit={(v) => updateSettings({ missing_threshold: v })} />
        <Field label="Doubtful from score" hint="listed as doubtful (not counted) from this score up to the missing score" value={st.doubt_threshold} min={0.01} max={0.9} onCommit={(v) => updateSettings({ doubt_threshold: v })} />
      </Card>

      <Card>
        <H>Alignment</H>
        <Field label="Min. inliers" hint="below this: RETAKE (cannot align the photo with the golden reference)" value={st.min_inliers} min={10} max={300} onCommit={(v) => updateSettings({ min_inliers: v })} />
        <Field label="Min. inlier ratio" value={st.min_inlier_ratio} min={0.05} max={0.9} onCommit={(v) => updateSettings({ min_inlier_ratio: v })} />
      </Card>

      {model ? (
        <Card>
          <H>Model</H>
          <Muted>
            {model.name}: {model.architecture}, trained {model.trained}. Exposure normalisation {model.exposure_norm ? 'on' : 'off'}.
          </Muted>
          {model.cross_validation_thorough ? (
            <Muted>
              Test on whole boards the model had never seen (5 folds), thorough mode: {model.cross_validation_thorough.missed} missing parts missed, {model.cross_validation_thorough.false_calls}{' '}
              good parts called missing; good photos wrongly failed {model.cross_validation_thorough.good_photos_failing}; removed parts found exactly in{' '}
              {model.cross_validation_thorough.removed_photos_exact} photos of boards with parts removed.
            </Muted>
          ) : model.cross_validation ? (
            <Muted>
              Test on whole boards the model had never seen ({model.cross_validation.folds} folds): {model.cross_validation.missing_crops - model.cross_validation.missed_at_threshold} of{' '}
              {model.cross_validation.missing_crops} missing parts found, {model.cross_validation.false_calls_at_threshold} false alarm in {model.cross_validation.present_crops} good parts.
            </Muted>
          ) : null}
          <Muted>Proven part types: {model.proven_part_types.join(', ') || 'none'}. Not yet proven: {model.not_proven_part_types.join(', ')}.</Muted>
          {model.note ? <Muted>{model.note}</Muted> : null}
        </Card>
      ) : null}

      <Btn
        title="Reset to defaults"
        kind="danger"
        onPress={() =>
          Alert.alert('Reset settings', 'Restore all settings to the defaults?', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Reset', style: 'destructive', onPress: () => void resetSettings() },
          ])
        }
      />
      <Muted style={{ textAlign: 'center' }}>
        Defaults: missing from score {DEFAULTS.missing_threshold}, doubtful from {DEFAULTS.doubt_threshold}
      </Muted>
    </ScrollView>
  );
}

const s = StyleSheet.create({
  page: { padding: 16, gap: 12, paddingBottom: 40 },
  field: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 4 },
  label: { color: C.text, fontSize: 15, fontWeight: '600' },
  input: {
    width: 84,
    height: 40,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: C.border,
    backgroundColor: C.card2,
    color: C.text,
    textAlign: 'right',
    paddingHorizontal: 10,
    fontSize: 15,
  },
});
