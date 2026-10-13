// Config plugin: bundle the JavaScript into the DEBUG apk too.
//
// React Native's Gradle plugin skips bundling for "debuggable variants" (debug) because a debug
// build normally loads JS from the Metro dev server. This app must run fully offline on the phone,
// so the debug variant is made non-debuggable for bundling purposes: app-debug.apk then contains
// the (production) Hermes bundle and starts without Metro.
const { withAppBuildGradle } = require('expo/config-plugins');

module.exports = function withBundledJs(config) {
  return withAppBuildGradle(config, (cfg) => {
    let src = cfg.modResults.contents;
    if (src.includes('debuggableVariants = []')) return cfg;
    const commented = /^\s*\/\/\s*debuggableVariants\s*=.*$/m;
    if (commented.test(src)) {
      src = src.replace(commented, '    debuggableVariants = []   // added by plugins/withBundledJs.js');
    } else {
      src = src.replace(/react\s*\{/, 'react {\n    debuggableVariants = []   // added by plugins/withBundledJs.js');
    }
    cfg.modResults.contents = src;
    return cfg;
  });
};
