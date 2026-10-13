package expo.modules.goldencomparator

import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import expo.modules.kotlin.Promise
import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.functions.Queues
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.File

/**
 * JS API of the golden-reference comparison (see modules/golden-comparator/index.ts).
 * All work functions are AsyncFunctions: they run on the Expo modules queue, never on the UI thread.
 * Paths are file:// URIs or absolute paths. Results are JSON strings.
 */
class GoldenComparatorModule : Module() {
  private val context get() = appContext.reactContext ?: throw CodedException("E_NO_CONTEXT", "No React context", null)

  private var presenceModel: PresenceModel? = null

  /** The ONNX part classifier + its model card from the APK assets (copied to the app's files folder once per process). */
  @Synchronized
  private fun loadModel(): PresenceModel {
    presenceModel?.let { return it }
    val dir = File(context.filesDir, "model").also { it.mkdirs() }
    val onnx = File(dir, "presence_cls.onnx")
    context.assets.open("presence_cls.onnx").use { input -> onnx.outputStream().use { output -> input.copyTo(output) } }
    val card = context.assets.open("presence_model.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
    return PresenceModel.load(onnx.path, card).also { presenceModel = it }
  }

  override fun definition() = ModuleDefinition {
    Name("GoldenComparator")
    Events("onProgress")

    OnCreate {
      if (!OpenCVLoader.initLocal()) throw CodedException("E_OPENCV", "OpenCV native library failed to load", null)
    }

    Function("defaultParams") { Params().toJson().toString() }

    /** Copies the bundled golden reference photo (Android asset golden_reference.jpg) to destUri -> its absolute path. */
    AsyncFunction("installBundledGolden") { destUri: String ->
      val dest = File(ImageIO.toPath(destUri))
      dest.parentFile?.mkdirs()
      context.assets.open("golden_reference.jpg").use { input ->
        dest.outputStream().use { output -> input.copyTo(output) }
      }
      dest.path
    }

    /**
     * Rectifies the photo (step A) and prepares the reference (step D): writes canonical.png, regions.json and
     * outline.jpg into outDir. regionsJson = regions to keep (user-edited / bundled names); null = detect them.
     * -> {ok, regions_count, hue, full_frame, sharpness, outline, image_size, regions} or {ok:false, retake}.
     */
    AsyncFunction("prepareReference") { imageUri: String, regionsJson: String?, paramsJson: String?, outDir: String ->
      val p = Params.fromJson(paramsJson)
      val photo = ImageIO.readBgr(context, imageUri)
      val out = JSONObject()
      try {
        val regions = regionsJson?.let { GoldenComparator.parseRegions(JSONObject(it), p) }
        val (board, ref) = GoldenComparator.prepareReference(photo, p, regions)
        File(ImageIO.toPath(outDir)).mkdirs()
        ImageIO.writePng("$outDir/canonical.png", board.canonical)
        ImageIO.writePng("$outDir/structure.png", ref.structure)   // what the local alignment of a part is measured on
        val rj = GoldenComparator.regionsJson(ref, p)
        ImageIO.writeText("$outDir/regions.json", rj.toString(1))
        ImageIO.writeJpg("$outDir/outline.jpg", outlinePreview(photo, board.outline), 85)
        out.put("ok", true)
          .put("regions_count", ref.regions.size)
          .put("hue", board.hue)
          .put("full_frame", board.fullFrame)
          .put("sharpness", board.sharpness)
          .put("outline", JSONArray(board.outline.toList()))
          .put("image_size", JSONArray(listOf(photo.cols(), photo.rows())))
          .put("regions", rj)
      } catch (e: Retake) {
        out.put("ok", false).put("retake", e.message)
      }
      out.toString()
    }

    /** The model card (presence_model.json from the APK assets) as JSON: name, thresholds, exposure normalisation, cross-validation numbers. */
    AsyncFunction("modelInfo") { loadModel().card.json.toString() }

    /**
     * The missing-part check: compares the photo with the saved reference (referenceDir/canonical.png + regions.json +
     * structure.png) and writes result.json, aligned.jpg and annotated.jpg into outDir (the photo itself is only read).
     * -> result JSON (decision PASS / FAIL / RETAKE, the missing and doubtful parts with their scores).
     */
    AsyncFunction("inspect") { referenceDir: String, imageUri: String, outDir: String, paramsJson: String? ->
      val p = Params.fromJson(paramsJson)
      val canonical = ImageIO.readBgrFile("$referenceDir/canonical.png")
      if (canonical.cols() != p.canonW || canonical.rows() != p.canonH) {
        throw CodedException(
          "E_REFERENCE_SIZE",
          "The saved reference is ${canonical.cols()}x${canonical.rows()} px but the check needs " +
            "${p.canonW}x${p.canonH} px. Rebuild the reference (restart the app).",
          null
        )
      }
      val regions = GoldenComparator.parseRegions(JSONObject(ImageIO.readText("$referenceDir/regions.json")), p)
      val structure = ImageIO.readGrayFile("$referenceDir/structure.png")
      val ref = Presence.lightReference(canonical, regions, structure, p)
      val photo = ImageIO.readBgr(context, imageUri)
      val res = Presence.inspect(ref, photo, p, loadModel()) { stage -> sendEvent("onProgress", mapOf("stage" to stage)) }
      File(ImageIO.toPath(outDir)).mkdirs()
      res.aligned?.let { ImageIO.writeJpg("$outDir/aligned.jpg", it, 90) }
      res.annotated?.let { ImageIO.writeJpg("$outDir/annotated.jpg", it, 92) }
      val text = res.result.toString(1)
      ImageIO.writeText("$outDir/result.json", text)
      text
    }

    View(GoldenCameraView::class) {
      Events("onCameraReady", "onCameraError")

      Prop("active") { view: GoldenCameraView, active: Boolean -> view.active = active }

      OnViewDidUpdateProps { view: GoldenCameraView -> view.recreateCamera() }

      OnViewDestroys { view: GoldenCameraView -> view.cleanup() }

      AsyncFunction("takePicture") { view: GoldenCameraView, path: String, promise: Promise ->
        view.takePicture(path, promise)
      }.runOnQueue(Queues.MAIN)

      AsyncFunction("focusAt") { view: GoldenCameraView, x: Double, y: Double ->
        view.focusAt(x.toFloat(), y.toFloat())
      }.runOnQueue(Queues.MAIN)
    }
  }

  /** The photo with the detected board outline, downscaled to <= 1200 px (golden reference confirmation). */
  private fun outlinePreview(photo: Mat, outline: DoubleArray): Mat {
    val s = minOf(1.0, 1200.0 / maxOf(photo.cols(), photo.rows()))
    val img = Mat()
    Imgproc.resize(photo, img, Size(), s, s, Imgproc.INTER_AREA)
    val pts = (0 until 4).map { Point(outline[it * 2] * s, outline[it * 2 + 1] * s) }
    Imgproc.polylines(img, listOf(MatOfPoint(*pts.map { Point(Math.rint(it.x), Math.rint(it.y)) }.toTypedArray())), true,
      Scalar(0.0, 255.0, 0.0), 4)
    for (c in pts) Imgproc.circle(img, c, 10, Scalar(0.0, 200.0, 255.0), -1)
    return img
  }
}
