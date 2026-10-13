#!/usr/bin/env node
// One-command build of the debug APK:   npm run build:apk
//
//  1. expo prebuild (generates android/ from app.json + the local module modules/golden-comparator)
//  2. gradlew assembleDebug   (JAVA_HOME = Android Studio's bundled JDK "jbr", SDK = %LOCALAPPDATA%\Android\Sdk)
//  3. copies the APK to android_app/app-debug.apk (and to android_app/app/build/outputs/apk/debug/)
//
// Options:  --clean   delete android/ first and prebuild from scratch
const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const root = path.resolve(__dirname, '..');
const win = process.platform === 'win32';
const clean = process.argv.includes('--clean');

function firstExisting(candidates) {
  return candidates.filter(Boolean).find((p) => fs.existsSync(p));
}

const jdk = firstExisting([
  process.env.JAVA_HOME && fs.existsSync(path.join(process.env.JAVA_HOME, 'bin')) ? process.env.JAVA_HOME : null,
  'C:\\Program Files\\Android\\Android Studio\\jbr',
  '/Applications/Android Studio.app/Contents/jbr/Contents/Home',
  '/opt/android-studio/jbr',
]);
const sdk = firstExisting([
  process.env.ANDROID_HOME,
  process.env.ANDROID_SDK_ROOT,
  process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Android', 'Sdk'),
  path.join(os.homedir(), 'Android', 'Sdk'),
  path.join(os.homedir(), 'Library', 'Android', 'sdk'),
]);
if (!jdk) throw new Error('No JDK found. Install Android Studio or set JAVA_HOME to its "jbr" folder.');
if (!sdk) throw new Error('No Android SDK found. Set ANDROID_HOME.');

const env = {
  ...process.env,
  JAVA_HOME: jdk,
  ANDROID_HOME: sdk,
  ANDROID_SDK_ROOT: sdk,
  PATH: [path.join(jdk, 'bin'), path.join(sdk, 'platform-tools'), process.env.PATH].join(path.delimiter),
};

// Windows workaround: on some machines the JDK's Unix-domain-socket selector wake-up fails inside
// %LOCALAPPDATA% ("java.io.IOException: Unable to establish loopback connection", which stops the Gradle
// daemon from starting). Pointing the socket directory at the Gradle home avoids it; harmless elsewhere.
if (win) {
  const sockDir = process.env.GRADLE_USER_HOME || path.join(os.homedir(), '.gradle');
  fs.mkdirSync(sockDir, { recursive: true });
  const opt = `-Djdk.net.unixdomain.tmpdir=${/\s/.test(sockDir) ? `"${sockDir}"` : sockDir}`;
  env.JAVA_TOOL_OPTIONS = [opt, process.env.JAVA_TOOL_OPTIONS].filter(Boolean).join(' ');
}

// With shell:true (needed for .bat files on Windows) arguments are concatenated, so quote the ones with spaces.
const quote = (a) => (win && /[\s"]/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a);

function run(cmd, args, cwd) {
  console.log(`\n> ${cmd} ${args.join(' ')}`);
  const r = spawnSync(cmd, args.map(quote), { cwd, env, stdio: 'inherit', shell: win });
  if (r.status !== 0) {
    console.error(`\nFAILED (${r.status}): ${cmd} ${args.join(' ')}`);
    process.exit(r.status || 1);
  }
}

const androidDir = path.join(root, 'android');
const gradlew = win ? '.\\gradlew.bat' : './gradlew';

// Developer helper:  node scripts/build-apk.js --gradle <any gradle arguments>   (same JDK / SDK / env as the build)
const passthrough = process.argv.indexOf('--gradle');
if (passthrough >= 0) {
  fs.writeFileSync(path.join(androidDir, 'local.properties'), `sdk.dir=${sdk.replace(/\\/g, '\\\\')}\n`);
  run(gradlew, process.argv.slice(passthrough + 1), androidDir);
  process.exit(0);
}

if (clean && fs.existsSync(androidDir)) fs.rmSync(androidDir, { recursive: true, force: true });
if (!fs.existsSync(path.join(androidDir, 'gradlew' + (win ? '.bat' : '')))) {
  run('npx', ['expo', 'prebuild', '--platform', 'android', '--no-install'], root);
}
fs.writeFileSync(path.join(androidDir, 'local.properties'), `sdk.dir=${sdk.replace(/\\/g, '\\\\')}\n`);

run(gradlew, ['assembleDebug'], androidDir);

const apk = path.join(androidDir, 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk');
if (!fs.existsSync(apk)) throw new Error(`APK not found: ${apk}`);
const targets = [
  path.join(root, 'app-debug.apk'),
  path.join(root, 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk'),
];
for (const t of targets) {
  fs.mkdirSync(path.dirname(t), { recursive: true });
  fs.copyFileSync(apk, t);
}
console.log(`\nAPK ready (${(fs.statSync(apk).size / 1048576).toFixed(1)} MB):\n  ${targets.join('\n  ')}`);
console.log('Install: adb install -r app-debug.apk   (or copy it to the phone and open it)');
