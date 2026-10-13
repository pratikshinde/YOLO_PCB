// Desktop JVM harness for the Kotlin pipeline: compiles the app's GoldenComparator.kt / Params.kt UNCHANGED
// (they only use org.opencv + org.json) against desktop OpenCV, so that the Kotlin port can be compared with
// scripts/golden_compare.py on the same photos without a phone.
//
//   node ../../scripts/build-apk.js --gradle -p tools/parity run --args="..."     (see README: "Parity test")
plugins {
  kotlin("jvm") version "2.1.20"
  application
}

repositories {
  mavenCentral()
}

dependencies {
  implementation("org.openpnp:opencv:4.9.0-0")   // desktop OpenCV 4.9 with the same Java API as org.opencv:opencv:4.10.0
  implementation("org.json:json:20240303")
}

sourceSets {
  main {
    kotlin {
      srcDir("src/main/kotlin")
      srcDir("../../modules/golden-comparator/android/src/main/java/expo/modules/goldencomparator")
      exclude("**/ImageIO.kt", "**/GoldenCameraView.kt", "**/GoldenComparatorModule.kt")
    }
  }
}

application {
  mainClass.set("ParityKt")
  applicationDefaultJvmArgs = listOf("-Xmx3g")
}
