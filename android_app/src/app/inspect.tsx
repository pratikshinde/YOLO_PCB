import { router, Stack } from 'expo-router';
import { useState } from 'react';
import { ActivityIndicator, Alert, Modal, ScrollView, StyleSheet, Text, View } from 'react-native';

import { CameraCapture } from '../components/CameraCapture';
import { Btn, PhotoTips } from '../components/ui';
import { runInspection } from '../lib/inspection';
import { C } from '../lib/theme';

export default function Inspect() {
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState('Starting');
  const [tips, setTips] = useState(false);

  const compare = async (uri: string, source: 'camera' | 'gallery') => {
    setRunning(true);
    setStage('Starting');
    try {
      const id = await runInspection(uri, source, setStage);
      router.replace({ pathname: '/result/[id]', params: { id } });
    } catch (e) {
      setRunning(false);
      Alert.alert('Check failed', String((e as Error)?.message ?? e));
    }
  };

  return (
    <View style={s.root}>
      <Stack.Screen options={{ headerRight: () => <Btn small kind="ghost" title="Photo tips" onPress={() => setTips(true)} /> }} />
      <CameraCapture onPicked={compare} />

      <Modal visible={tips} transparent animationType="fade" onRequestClose={() => setTips(false)}>
        <View style={s.modalBack}>
          <ScrollView contentContainerStyle={{ padding: 16, gap: 12 }}>
            <PhotoTips />
            <Btn title="Close" onPress={() => setTips(false)} />
          </ScrollView>
        </View>
      </Modal>

      {running ? (
        <View style={s.overlay}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={s.big}>Checking components...</Text>
          <Text style={s.stage}>{stage}</Text>
        </View>
      ) : null}
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  modalBack: { flex: 1, backgroundColor: 'rgba(0,0,0,0.88)', justifyContent: 'center' },
  overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.8)', alignItems: 'center', justifyContent: 'center', gap: 10 },
  big: { color: '#fff', fontSize: 22, fontWeight: '800' },
  stage: { color: C.muted, fontSize: 14 },
});
