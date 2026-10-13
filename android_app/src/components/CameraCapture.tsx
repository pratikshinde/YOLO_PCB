import { useFocusEffect } from 'expo-router';
import { File, Paths } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, PermissionsAndroid, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { GoldenCameraView, type GoldenCameraHandle } from '../../modules/golden-comparator';
import { C } from '../lib/theme';
import { Btn } from './ui';

type Props = {
  onPicked: (uri: string, source: 'camera' | 'gallery') => void;
  /** shown when a photo is imported from the gallery (e.g. a warning for the golden reference) */
  galleryWarning?: string;
  hint?: string;
};

const BOARD_ASPECT = 83 / 116; // portrait guide frame, like the golden reference

/** Back camera with a board-shaped guide frame, tap-to-focus, shutter and gallery import. */
export function CameraCapture({ onPicked, galleryWarning, hint }: Props) {
  const cam = useRef<GoldenCameraHandle>(null);
  const insets = useSafeAreaInsets();
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [permission, setPermission] = useState<'unknown' | 'granted' | 'denied'>('unknown');
  const [focused, setFocused] = useState(true);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [ring, setRing] = useState<{ x: number; y: number; k: number } | null>(null);

  useFocusEffect(
    useCallback(() => {
      setFocused(true);
      return () => setFocused(false);
    }, [])
  );

  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'android') return setPermission('granted');
      const have = await PermissionsAndroid.check(PermissionsAndroid.PERMISSIONS.CAMERA);
      if (have) return setPermission('granted');
      const r = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA);
      setPermission(r === PermissionsAndroid.RESULTS.GRANTED ? 'granted' : 'denied');
    })();
  }, []);

  // camera area: 3:4 portrait (the 4:3 sensor image turned), as large as fits
  const camH = Math.min(box.h, (box.w * 4) / 3);
  const camW = (camH * 3) / 4;
  let gw = camW * 0.84;
  let gh = gw / BOARD_ASPECT;
  if (gh > camH * 0.94) {
    gh = camH * 0.94;
    gw = gh * BOARD_ASPECT;
  }

  const shoot = async () => {
    if (!cam.current || busy) return;
    setBusy(true);
    try {
      const { uri } = await cam.current.takePicture(new File(Paths.cache, `capture-${Date.now()}.jpg`).uri);
      onPicked(uri, 'camera');
    } catch (e) {
      Alert.alert('Capture failed', String((e as Error)?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const pickFromGallery = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1, allowsEditing: false });
    if (res.canceled || !res.assets?.[0]) return;
    const uri = res.assets[0].uri;
    if (galleryWarning) {
      Alert.alert('Imported photo', galleryWarning, [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Use this photo', onPress: () => onPicked(uri, 'gallery') },
      ]);
    } else {
      onPicked(uri, 'gallery');
    }
  };

  const onTap = (x: number, y: number) => {
    cam.current?.focusAt(x, y).catch(() => {});
    setRing({ x, y, k: Date.now() });
    setTimeout(() => setRing((r) => (r && Date.now() - r.k >= 1200 ? null : r)), 1300);
  };

  return (
    <View style={s.root}>
      <View style={s.stage} onLayout={(e) => setBox({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}>
        {permission === 'granted' && camW > 0 ? (
          // no overflow:hidden / borderRadius here: clipping a camera surface can make it render black on Android
          <View style={{ width: camW, height: camH }}>
            <GoldenCameraView
              ref={cam}
              style={{ width: camW, height: camH }}
              active={focused}
              onCameraReady={() => {
                setReady(true);
                setError(null);
              }}
              onCameraError={(e) => setError(e.nativeEvent.message)}
            />
            <Pressable
              style={StyleSheet.absoluteFill}
              onPress={(e) => {
                const ne = e.nativeEvent as { locationX?: number; locationY?: number; offsetX?: number; offsetY?: number };
                onTap(ne.locationX ?? ne.offsetX ?? 0, ne.locationY ?? ne.offsetY ?? 0);
              }}>
              <View pointerEvents="none" style={[s.guide, { width: gw, height: gh, left: (camW - gw) / 2, top: (camH - gh) / 2 }]}>
                <View style={[s.corner, { top: -2, left: -2, borderTopWidth: 4, borderLeftWidth: 4 }]} />
                <View style={[s.corner, { top: -2, right: -2, borderTopWidth: 4, borderRightWidth: 4 }]} />
                <View style={[s.corner, { bottom: -2, left: -2, borderBottomWidth: 4, borderLeftWidth: 4 }]} />
                <View style={[s.corner, { bottom: -2, right: -2, borderBottomWidth: 4, borderRightWidth: 4 }]} />
                <Text style={s.guideText}>Fit the whole board in the frame</Text>
                <Text style={s.guideText}>(header pins at the bottom)</Text>
              </View>
              {ring ? <View pointerEvents="none" style={[s.ring, { left: ring.x - 28, top: ring.y - 28 }]} /> : null}
            </Pressable>
            {!ready && !error ? <ActivityIndicator style={StyleSheet.absoluteFill} color={C.text} /> : null}
          </View>
        ) : (
          <View style={s.center}>
            {permission === 'denied' ? (
              <Text style={s.msg}>Camera permission was denied. Allow it in the Android settings, or import a photo from the gallery.</Text>
            ) : (
              <ActivityIndicator color={C.text} />
            )}
          </View>
        )}
        {error ? <Text style={[s.msg, s.err]}>{error}</Text> : null}
      </View>

      <View style={[s.bar, { paddingBottom: 12 + insets.bottom }]}>
        <View style={s.barSide}>
          <Btn title="Gallery" onPress={pickFromGallery} style={{ minWidth: 96 }} />
        </View>
        <Pressable
          accessibilityLabel="Take photo"
          onPress={shoot}
          disabled={permission !== 'granted' || !ready || busy}
          style={[s.shutter, (permission !== 'granted' || !ready || busy) && { opacity: 0.4 }]}>
          {busy ? <ActivityIndicator color="#000" /> : <View style={s.shutterInner} />}
        </Pressable>
        <View style={[s.barSide, { alignItems: 'flex-end' }]}>
          <Text style={s.hint}>{hint ?? 'Tap the board to focus'}</Text>
        </View>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  stage: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  center: { alignItems: 'center', justifyContent: 'center', padding: 24 },
  guide: { position: 'absolute', borderWidth: 1, borderColor: 'rgba(255,255,255,0.55)', justifyContent: 'flex-end', alignItems: 'center' },
  corner: { position: 'absolute', width: 26, height: 26, borderColor: '#fff' },
  guideText: { color: 'rgba(255,255,255,0.85)', fontSize: 11, textAlign: 'center', marginBottom: 8, textShadowColor: '#000', textShadowRadius: 3 },
  ring: { position: 'absolute', width: 56, height: 56, borderRadius: 28, borderWidth: 2, borderColor: C.amber },
  msg: { color: C.text, textAlign: 'center', fontSize: 14 },
  err: { position: 'absolute', bottom: 8, left: 12, right: 12, color: C.amber },
  bar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12, backgroundColor: C.bg },
  barSide: { flex: 1, justifyContent: 'center' },
  shutter: { width: 72, height: 72, borderRadius: 36, borderWidth: 4, borderColor: '#fff', alignItems: 'center', justifyContent: 'center' },
  shutterInner: { width: 54, height: 54, borderRadius: 27, backgroundColor: '#fff' },
  hint: { color: C.muted, fontSize: 11, textAlign: 'right' },
});
