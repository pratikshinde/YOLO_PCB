import expo.modules.goldencomparator.GoldenComparator
import expo.modules.goldencomparator.Params
import expo.modules.goldencomparator.Presence
import expo.modules.goldencomparator.PresenceModel
import org.json.JSONObject
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * Desktop check of the Kotlin missing-part pipeline (Presence.kt + the alignment of GoldenComparator.kt) against the Python
 * reference (ml/presence/predict_photo.py). For every photo it writes <out>/<photo>.json (the same result JSON the app writes) and
 * <out>/<photo>.annotated.jpg and prints decision, missing parts and time.
 *
 *   args: <golden.jpg> <regions.json> <model.onnx> <presence_model.json> <outDir> <photo> [<photo> ...]
 */
fun main(args: Array<String>) {
  require(args.size >= 6) { "usage: golden.jpg regions.json model.onnx presence_model.json outDir photo..." }
  nu.pattern.OpenCV.loadLocally()
  val goldenPath = args[0]
  val regionsPath = args[1]
  val onnxPath = args[2]
  val cardPath = args[3]
  val outDir = args[4]
  val card = JSONObject(File(cardPath).readText())
  val p = Params()
  p.missingThreshold = card.getDouble("threshold")
  p.doubtThreshold = card.getDouble("doubt")
  val regions = GoldenComparator.parseRegions(JSONObject(File(regionsPath).readText()), p)
  val t0 = System.currentTimeMillis()
  val (_, full) = GoldenComparator.prepareReference(Imgcodecs.imread(goldenPath), p, regions)
  // the app inspects with the light reference (canonical + structure only): do the same here
  val ref = Presence.lightReference(full.canonical, regions, full.structure, p)
  println("reference prepared in %.1f s".format((System.currentTimeMillis() - t0) / 1000.0))
  val model = PresenceModel.load(onnxPath, card.toString())
  File(outDir).mkdirs()
  for (photo in args.drop(5)) {
    val img = Imgcodecs.imread(photo)
    if (img.empty()) {
      println("${File(photo).name}: cannot read")
      continue
    }
    val res = Presence.inspect(ref, img, p, model)
    val stem = File(photo).nameWithoutExtension
    File(outDir, "$stem.json").writeText(res.result.toString(1))
    res.annotated?.let { Imgcodecs.imwrite(File(outDir, "$stem.annotated.jpg").path, it) }
    val r = res.result
    val missing = r.getJSONArray("missing")
    val names = (0 until missing.length()).joinToString(", ") { missing.getJSONObject(it).getString("name") + " " + missing.getJSONObject(it).getDouble("p_missing") }
    println("${File(photo).name}: ${r.getString("decision")} - ${r.getString("reason").take(120)}  [$names]  ${r.getDouble("time_s")} s")
  }
}
