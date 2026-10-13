import { memo } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';

import type { Box } from '../../modules/golden-comparator';

export type BoxSpec = {
  key: string;
  box: Box; // mm, board frame
  color: string;
  width?: number;
  label?: string;
  fill?: string;
  onPress?: () => void;
};

export type ArrowSpec = { key: string; from: [number, number]; to: [number, number]; color: string };

type Props = {
  uri: string;
  boardMm: [number, number]; // width, height (portrait: 83 x 116)
  widthPx: number;
  boxes?: BoxSpec[];
  arrows?: ArrowSpec[];
  dim?: number; // 0..1 darkening of the image so that boxes stand out
};

/** A board image (canonical frame, so 1 mm = widthPx / boardMm[0] px) with boxes / arrows drawn in mm. */
function BoardViewImpl({ uri, boardMm, widthPx, boxes = [], arrows = [], dim = 0 }: Props) {
  const k = widthPx / boardMm[0];
  const height = boardMm[1] * k;
  return (
    <View style={{ width: widthPx, height, backgroundColor: '#000' }}>
      <Image source={{ uri }} style={{ width: widthPx, height }} resizeMode="stretch" fadeDuration={0} />
      {dim > 0 ? <View pointerEvents="none" style={[StyleSheet.absoluteFill, { backgroundColor: `rgba(0,0,0,${dim})` }]} /> : null}
      {boxes.map((b) => {
        const style = {
          position: 'absolute' as const,
          left: b.box[0] * k,
          top: b.box[1] * k,
          width: Math.max(b.box[2] * k, 2),
          height: Math.max(b.box[3] * k, 2),
          borderWidth: b.width ?? 2,
          borderColor: b.color,
          backgroundColor: b.fill,
        };
        const label = b.label ? (
          <Text numberOfLines={1} style={[s.label, { color: b.color, top: -11 }]}>
            {b.label}
          </Text>
        ) : null;
        return b.onPress ? (
          <Pressable key={b.key} style={style} onPress={b.onPress}>
            {label}
          </Pressable>
        ) : (
          <View key={b.key} style={style} pointerEvents="none">
            {label}
          </View>
        );
      })}
      {arrows.map((a) => {
        const x1 = a.from[0] * k;
        const y1 = a.from[1] * k;
        const x2 = a.to[0] * k;
        const y2 = a.to[1] * k;
        const len = Math.hypot(x2 - x1, y2 - y1);
        const ang = Math.atan2(y2 - y1, x2 - x1);
        return (
          <View
            key={a.key}
            pointerEvents="none"
            style={{
              position: 'absolute',
              left: (x1 + x2) / 2 - len / 2,
              top: (y1 + y2) / 2 - 1,
              width: len,
              height: 2,
              backgroundColor: a.color,
              transform: [{ rotate: `${ang}rad` }],
            }}
          />
        );
      })}
    </View>
  );
}

export const BoardView = memo(BoardViewImpl);

const s = StyleSheet.create({
  label: {
    position: 'absolute',
    left: -1,
    fontSize: 9,
    fontWeight: '700',
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 2,
  },
});
