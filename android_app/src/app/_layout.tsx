import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { Btn } from '../components/ui';
import { getSettings, loadSettings } from '../lib/settings';
import { ensureDefaultReference } from '../lib/storage';
import { C } from '../lib/theme';

export default function RootLayout() {
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    (async () => {
      try {
        setState('loading');
        loadSettings();
        // first start: install the bundled golden reference (rectify + regions), a few seconds
        await ensureDefaultReference(getSettings());
        setState('ready');
      } catch (e) {
        setError(String((e as Error)?.message ?? e));
        setState('error');
      }
    })();
  }, [attempt]);

  if (state !== 'ready') {
    return (
      <View style={s.splash}>
        <StatusBar style="light" />
        <Text style={s.title}>Missing Part Check</Text>
        {state === 'loading' ? (
          <>
            <ActivityIndicator color={C.text} style={{ marginTop: 16 }} />
            <Text style={s.sub}>Preparing the golden reference (first start only, about a minute)...</Text>
          </>
        ) : (
          <>
            <Text style={[s.sub, { color: C.amber }]}>{error}</Text>
            <Btn title="Try again" onPress={() => setAttempt((a) => a + 1)} style={{ marginTop: 16 }} />
          </>
        )}
      </View>
    );
  }

  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: C.bg },
          headerTintColor: C.text,
          headerTitleStyle: { fontWeight: '700' },
          contentStyle: { backgroundColor: C.bg },
        }}>
        <Stack.Screen name="index" options={{ title: 'Missing Part Check' }} />
        <Stack.Screen name="inspect" options={{ title: 'Inspect new board' }} />
        <Stack.Screen name="result/[id]" options={{ title: 'Result' }} />
        <Stack.Screen name="history" options={{ title: 'History' }} />
        <Stack.Screen name="settings" options={{ title: 'Settings' }} />
      </Stack>
    </>
  );
}

const s = StyleSheet.create({
  splash: { flex: 1, backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { color: C.text, fontSize: 24, fontWeight: '800' },
  sub: { color: C.muted, marginTop: 12, textAlign: 'center' },
});
