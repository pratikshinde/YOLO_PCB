import type { ReactNode } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View, type StyleProp, type TextStyle, type ViewStyle } from 'react-native';

import { C } from '../lib/theme';

type BtnProps = {
  title: string;
  onPress?: () => void;
  kind?: 'primary' | 'secondary' | 'danger' | 'ghost';
  disabled?: boolean;
  busy?: boolean;
  small?: boolean;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
};

export function Btn({ title, onPress, kind = 'secondary', disabled, busy, small, style, textStyle }: BtnProps) {
  const bg = { primary: C.accent, secondary: C.card2, danger: '#7f1d1d', ghost: 'transparent' }[kind];
  const border = kind === 'ghost' ? C.border : kind === 'secondary' ? C.border : 'transparent';
  return (
    <Pressable
      accessibilityRole="button"
      onPress={disabled || busy ? undefined : onPress}
      style={({ pressed }) => [
        s.btn,
        small && s.btnSmall,
        { backgroundColor: bg, borderColor: border, opacity: disabled ? 0.4 : pressed ? 0.8 : 1 },
        style,
      ]}>
      {busy ? <ActivityIndicator color={C.text} size="small" /> : null}
      <Text style={[s.btnText, small && { fontSize: 13 }, textStyle]}>{title}</Text>
    </Pressable>
  );
}

export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[s.card, style]}>{children}</View>;
}

export function H({ children, style }: { children: ReactNode; style?: StyleProp<TextStyle> }) {
  return <Text style={[s.h, style]}>{children}</Text>;
}

export function Muted({ children, style, numberOfLines }: { children: ReactNode; style?: StyleProp<TextStyle>; numberOfLines?: number }) {
  return (
    <Text style={[s.muted, style]} numberOfLines={numberOfLines}>
      {children}
    </Text>
  );
}

export function Chip({ label, color, active, onPress }: { label: string; color?: string; active?: boolean; onPress?: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      style={[s.chip, { borderColor: color ?? C.border, backgroundColor: active ? (color ?? C.accent) : 'transparent' }]}>
      <Text style={[s.chipText, active && { color: '#000', fontWeight: '700' }]}>{label}</Text>
    </Pressable>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T | string;
  onChange: (v: T) => void;
}) {
  return (
    <View style={s.seg}>
      {options.map((o) => (
        <Pressable key={o.value} onPress={() => onChange(o.value)} style={[s.segItem, value === o.value && s.segActive]}>
          <Text style={[s.segText, value === o.value && { color: '#fff', fontWeight: '700' }]}>{o.label}</Text>
        </Pressable>
      ))}
    </View>
  );
}

export const TIPS = [
  'Photograph the whole board straight from above, lying flat, with 1-3 cm of background all around it (the board must not touch the edge of the picture).',
  'Plain background that is not red (the red board must stand out). Soft, even light: no flash, no lamp shining straight onto the board.',
  'Tap the board on the preview to focus before you shoot, and hold the phone still: a blurred photo is rejected (RETAKE) and hides small parts.',
  'Red LG Indicator boards only. A board of another colour is not recognised.',
];

export function PhotoTips({ extra }: { extra?: string }) {
  return (
    <Card>
      <H style={{ fontSize: 15 }}>Photo tips</H>
      {TIPS.map((t) => (
        <Text key={t} style={s.tip}>
          {'•'} {t}
        </Text>
      ))}
      {extra ? <Text style={[s.tip, { color: C.amber }]}>{extra}</Text> : null}
    </Card>
  );
}

export function Row({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[{ flexDirection: 'row', alignItems: 'center', gap: 8 }, style]}>{children}</View>;
}

const s = StyleSheet.create({
  btn: {
    minHeight: 46,
    paddingHorizontal: 16,
    borderRadius: 10,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  btnSmall: { minHeight: 36, paddingHorizontal: 10, borderRadius: 8 },
  btnText: { color: C.text, fontSize: 16, fontWeight: '600' },
  card: { backgroundColor: C.card, borderRadius: 14, borderWidth: 1, borderColor: C.border, padding: 14, gap: 8 },
  h: { color: C.text, fontSize: 18, fontWeight: '700' },
  muted: { color: C.muted, fontSize: 13 },
  chip: { borderWidth: 1, borderRadius: 16, paddingHorizontal: 10, paddingVertical: 5 },
  chipText: { color: C.text, fontSize: 12 },
  seg: { flexDirection: 'row', backgroundColor: C.card2, borderRadius: 10, borderWidth: 1, borderColor: C.border, overflow: 'hidden' },
  segItem: { flex: 1, paddingVertical: 10, alignItems: 'center' },
  segActive: { backgroundColor: C.accent },
  segText: { color: C.muted, fontSize: 14 },
  tip: { color: C.muted, fontSize: 13, lineHeight: 19 },
});
