package expo.modules.goldencomparator

import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * What the part crops must look like: the model card presence_model.json written by ml/presence/v3/export_app_model.py.
 * Everything here has to be the same as in training, otherwise the scores mean nothing.
 */
class ModelCard(val json: JSONObject) {
  val name: String = json.optString("name", "presence_cls")
  val inputSize: Int = json.optInt("input_size", 128)
  val marginMm: Double = json.optDouble("crop_margin_mm", 1.5)
  val missingIndex: Int = json.optInt("missing_index", 0)
  val exposureNorm: Boolean = json.optBoolean("exposure_norm", false)
  val targetLuma: Double = json.optDouble("exposure_target_luma", 0.0)
  val gainMin: Double = json.optDouble("exposure_gain_min", 1.0)
  val gainMax: Double = json.optDouble("exposure_gain_max", 1.0)
  val threshold: Double = json.optDouble("threshold", 0.35)
  val doubt: Double = json.optDouble("doubt", 0.10)
  val maxFlagged: Int = json.optInt("max_flagged", 20)

  /** "Thorough" mode: every part is looked at through these windows (offsets in mm from the part's local position; the first is the centre) and the MEDIAN score counts. */
  val windowOffsetsMm: List<DoubleArray> = json.optJSONArray("window_offsets_mm")?.let { a ->
    (0 until a.length()).map { i -> doubleArrayOf(a.getJSONArray(i).getDouble(0), a.getJSONArray(i).getDouble(1)) }
  } ?: listOf(doubleArrayOf(0.0, 0.0))
}

/** The YOLO11n-cls part classifier (ONNX) run by OpenCV's DNN module: crop -> p(missing). */
class PresenceModel(private val net: Net, val card: ModelCard) {
  fun pMissing(crop: Mat): Double {
    val blob = Dnn.blobFromImage(crop, 1.0 / 255.0, Size(card.inputSize.toDouble(), card.inputSize.toDouble()), Scalar(0.0, 0.0, 0.0), true, false)
    net.setInput(blob)
    val out = net.forward()               // 1 x 2, softmax already applied: [p(missing), p(present)]
    val v = out.get(0, card.missingIndex)[0]
    blob.release()
    out.release()
    return v
  }

  companion object {
    fun load(onnxPath: String, cardJson: String): PresenceModel {
      val net = Dnn.readNetFromONNX(onnxPath)
      net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
      net.setPreferableTarget(Dnn.DNN_TARGET_CPU)
      return PresenceModel(net, ModelCard(JSONObject(cardJson)))
    }
  }
}

/** One part of the board: its score (median over the windows in thorough mode), the score of the centre window alone, and where the part was looked for. */
class PartScore(val region: Region, val p: Double, val pCentre: Double, val shift: IntArray)

/**
 * The missing-part check. Photo -> board found, rectified and aligned to the golden reference (20 px/mm) -> small local
 * shift of every part (optical flow) -> one 128x128 crop per part -> p(missing) from the model -> the board FAILS when a part
 * has p(missing) >= threshold. A copy of ml/presence/predict_photo.py (same steps, same numbers) - keep the two in step.
 */
object Presence {
  private fun mm(v: Int, p: Params): Double = round(v / p.pxPerMm * 100) / 100

  private fun boxMm(b: IntArray, p: Params) = JSONArray(listOf(mm(b[0], p), mm(b[1], p), mm(b[2], p), mm(b[3], p)))

  /** Reference prepared for inspection: everything the check needs, loaded from the saved files (no colour analysis any more). */
  fun lightReference(canonical: Mat, regions: List<Region>, structure: Mat, p: Params): Reference {
    val valid = Mat(canonical.rows(), canonical.cols(), CvType.CV_8U, Scalar(255.0))
    val area = GoldenComparator.boardArea(valid, p)
    val clahe = GoldenComparator.claheGray(canonical)
    val none = Mat()
    return Reference(canonical, valid, clahe, none, area, none, regions, none, none, none, structure, emptyMap())
  }

  /** Same as Python square_crop(): window of `side` px centred on (cx, cy), reflected at the image border, resized (INTER_AREA). */
  fun squareCrop(img: Mat, cx: Double, cy: Double, side: Int, out: Int): Mat {
    val h = img.rows()
    val w = img.cols()
    val half = side / 2.0
    val x1 = Math.rint(cx - half).toInt()
    val y1 = Math.rint(cy - half).toInt()
    val x2 = x1 + side
    val y2 = y1 + side
    val padTop = max(0, -y1)
    val padBottom = max(0, y2 - h)
    val padLeft = max(0, -x1)
    val padRight = max(0, x2 - w)
    var patch = img.submat(max(0, y1), min(h, y2), max(0, x1), min(w, x2))
    if (padTop > 0 || padBottom > 0 || padLeft > 0 || padRight > 0) {
      val padded = Mat()
      Core.copyMakeBorder(patch, padded, padTop, padBottom, padLeft, padRight, Core.BORDER_REFLECT_101)
      patch = padded
    }
    patch = patch.submat(0, min(side, patch.rows()), 0, min(side, patch.cols()))
    val resized = Mat()
    Imgproc.resize(patch, resized, Size(out.toDouble(), out.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
    return resized
  }

  /** Median of a few scores (numpy semantics: mean of the two middle values when the count is even). */
  fun median(v: DoubleArray): Double {
    if (v.size == 1) return v[0]
    val s = v.sortedArray()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2.0
  }

  /** Median of a 256-bin histogram (numpy semantics: mean of the two middle samples when the count is even). */
  private fun histMedian(hist: IntArray): Double {
    var n = 0
    for (c in hist) n += c
    fun kth(k: Int): Int {
      var acc = 0
      for (i in hist.indices) {
        acc += hist[i]
        if (acc > k) return i
      }
      return hist.size - 1
    }
    return if (n % 2 == 1) kth(n / 2).toDouble() else (kth(n / 2 - 1) + kth(n / 2)) / 2.0
  }

  /**
   * Exposure gain of a photo (the model was trained on exposure-normalised crops): median colour of the red solder-mask pixels
   * (HSV hue <= 10 or >= 170, saturation >= 120, value >= 50) -> luma -> target_luma / luma, limited to [gainMin, gainMax].
   */
  fun exposureGain(aligned: Mat, card: ModelCard): Double {
    val hsv = Mat()
    Imgproc.cvtColor(aligned, hsv, Imgproc.COLOR_BGR2HSV)
    val n = aligned.rows() * aligned.cols()
    val a = ByteArray(n * 3)
    val h = ByteArray(n * 3)
    (if (aligned.isContinuous) aligned else aligned.clone()).get(0, 0, a)
    (if (hsv.isContinuous) hsv else hsv.clone()).get(0, 0, h)
    val hb = IntArray(256)
    val hg = IntArray(256)
    val hr = IntArray(256)
    var count = 0
    for (i in 0 until n) {
      val hue = h[3 * i].toInt() and 0xff
      val s = h[3 * i + 1].toInt() and 0xff
      val v = h[3 * i + 2].toInt() and 0xff
      if ((hue <= 10 || hue >= 170) && s >= 120 && v >= 50) {
        hb[a[3 * i].toInt() and 0xff]++
        hg[a[3 * i + 1].toInt() and 0xff]++
        hr[a[3 * i + 2].toInt() and 0xff]++
        count++
      }
    }
    hsv.release()
    if (count < 20000) return 1.0
    val luma = 0.114 * histMedian(hb) + 0.587 * histMedian(hg) + 0.299 * histMedian(hr)
    return (card.targetLuma / luma).coerceIn(card.gainMin, card.gainMax)
  }

  /** Steps A + B; a board that touches the photo edge gets one more try with a grey margin added in memory (as in training). */
  private fun prepare(ref: Reference, bgr: Mat, p: Params, progress: (String) -> Unit): Triple<GoldenComparator.Prepared, Boolean, Mat> {
    try {
      return Triple(GoldenComparator.prepareNew(ref, bgr, p, progress), false, bgr)
    } catch (e: Retake) {
      if (e.message?.contains("edge") != true) throw e
      val padded = Mat()
      Core.copyMakeBorder(bgr, padded, 80, 80, 80, 80, Core.BORDER_CONSTANT, Scalar(128.0, 130.0, 128.0))
      return Triple(GoldenComparator.prepareNew(ref, padded, p, progress), true, padded)
    }
  }

  private fun partJson(s: PartScore, status: String, p: Params): JSONObject = JSONObject()
    .put("region_id", s.region.id)
    .put("name", s.region.name)
    .put("status", status)
    .put("p_missing", GoldenComparator.r3(s.p))
    .put("p_centre", GoldenComparator.r3(s.pCentre))
    .put("bbox_mm", boxMm(s.region.bbox, p))
    .put("centre_mm", JSONArray(listOf(round((s.region.bbox[0] + s.region.bbox[2] / 2.0 + s.shift[0]) / p.pxPerMm * 100) / 100,
      round((s.region.bbox[1] + s.region.bbox[3] / 2.0 + s.shift[1]) / p.pxPerMm * 100) / 100)))
    .put("shift_mm", JSONArray(listOf(mm(s.shift[0], p), mm(s.shift[1], p))))

  private fun counts(parts: Int, missing: Int, doubtful: Int) =
    JSONObject().put("parts", parts).put("missing", missing).put("doubtful", doubtful)

  fun inspect(ref: Reference, newBgr: Mat, p: Params, model: PresenceModel, progress: (String) -> Unit = {}): InspectionOutput {
    val t0 = System.currentTimeMillis()
    val card = model.card
    val base = JSONObject().put("params", p.toJson())
      .put("model", JSONObject().put("name", card.name).put("threshold", p.missingThreshold).put("doubt", p.doubtThreshold))
    fun retake(reason: String) = InspectionOutput(
      base.put("decision", "RETAKE").put("reason", reason).put("counts", counts(0, 0, 0))
        .put("missing", JSONArray()).put("doubtful", JSONArray()).put("parts", JSONArray())
        .put("time_s", (System.currentTimeMillis() - t0) / 1000.0),
      null, null)

    progress("Detecting the board")
    val (prep, marginAdded, _) = try {
      prepare(ref, newBgr, p, progress)
    } catch (e: Retake) {
      return retake(e.message ?: "retake")
    }
    val aligned = prep.aligned
    val al = prep.alignment

    val tPrepared = System.currentTimeMillis()
    progress("Local alignment of the parts")
    val (refGray, newGray, _) = GoldenComparator.comparisonGrays(ref.clahe, GoldenComparator.claheGray(aligned), prep.area, p)
    val shifts = GoldenComparator.regionShifts(GoldenComparator.localFlow(refGray, newGray), ref, ref.regions, p)
    val tLocal = System.currentTimeMillis()

    progress("Checking the parts")
    val gain = if (card.exposureNorm) exposureGain(aligned, card) else 1.0
    val source = if (gain != 1.0) Mat().also { aligned.convertTo(it, -1, gain, -0.5) } else aligned   // -0.5: truncation like numpy astype(uint8)
    val parts = ref.regions.filter { it.kind == "part" }
    val offsets = if (p.windowCount >= 2) card.windowOffsetsMm else listOf(doubleArrayOf(0.0, 0.0))
    val scores = ArrayList<PartScore>(parts.size)
    for (r in parts) {
      val sh = shifts[r.id] ?: intArrayOf(0, 0)
      val side = Math.rint(max(r.bbox[2], r.bbox[3]) + 2 * card.marginMm * p.pxPerMm).toInt()
      val cx = r.bbox[0] + r.bbox[2] / 2.0 + sh[0]
      val cy = r.bbox[1] + r.bbox[3] / 2.0 + sh[1]
      val ps = DoubleArray(offsets.size)
      for ((i, o) in offsets.withIndex()) {
        val crop = squareCrop(source, cx + o[0] * p.pxPerMm, cy + o[1] * p.pxPerMm, side, card.inputSize)
        ps[i] = model.pMissing(crop)
        crop.release()
      }
      scores.add(PartScore(r, median(ps), ps[0], sh))
    }
    if (source !== aligned) source.release()
    val tParts = System.currentTimeMillis()

    val ordered = scores.sortedByDescending { it.p }
    val missing = ordered.filter { it.p >= p.missingThreshold }
    val doubtful = ordered.filter { it.p >= p.doubtThreshold && it.p < p.missingThreshold }
    val decision: String
    val reason: String
    if (missing.size > card.maxFlagged) {
      return retake("${missing.size} of ${scores.size} parts look missing - that is not plausible: wrong board, a bad photo or a failed alignment. " +
        "Retake the photo (whole board in the frame, sharp, no strong reflections).")
    } else if (missing.isNotEmpty()) {
      decision = "FAIL"
      reason = missing.size.toString() + (if (missing.size == 1) " component missing: " else " components missing: ") +
        missing.joinToString(", ") { it.region.name }
    } else {
      decision = "PASS"
      reason = "No missing components (all ${scores.size} parts found)" +
        if (doubtful.isNotEmpty()) "; ${doubtful.size} doubtful: " + doubtful.take(5).joinToString(", ") { it.region.name } else ""
    }

    val result = base
      .put("decision", decision)
      .put("reason", reason)
      .put("counts", counts(scores.size, missing.size, doubtful.size))
      .put("missing", JSONArray(missing.map { partJson(it, "missing", p) }))
      .put("doubtful", JSONArray(doubtful.map { partJson(it, "doubtful", p) }))
      .put("parts", JSONArray(ordered.map { partJson(it, if (it.p >= p.missingThreshold) "missing" else if (it.p >= p.doubtThreshold) "doubtful" else "present", p) }))
      .put("alignment", JSONObject()
        .put("method", al.method).put("inliers", al.inliers).put("matches", al.matches)
        .put("inlier_ratio", GoldenComparator.r3(al.inlierRatio)).put("residual_mm", GoldenComparator.r3(al.residualMm))
        .put("rotation_deg", round(al.rotationDeg * 10) / 10).put("coverage", GoldenComparator.r3(prep.coverage))
        .put("full_frame_input", prep.board.fullFrame).put("sharpness", round(prep.board.sharpness * 10) / 10)
        .put("margin_added", marginAdded))
      .put("local_alignment", JSONObject()
        .put("regions_shifted", shifts.size)
        .put("max_shift_mm", round((shifts.values.maxOfOrNull { hypot(it[0].toDouble(), it[1].toDouble()) } ?: 0.0) / p.pxPerMm * 100) / 100))
      .put("exposure_gain", round(gain * 1000) / 1000)
      .put("windows", offsets.size)
      .put("timing", JSONObject()
        .put("prepare_s", (tPrepared - t0) / 1000.0).put("local_alignment_s", (tLocal - tPrepared) / 1000.0)
        .put("parts_s", (tParts - tLocal) / 1000.0).put("crops", scores.size * offsets.size))
      .put("time_s", (System.currentTimeMillis() - t0) / 1000.0)
    progress("Saving")
    return InspectionOutput(result, aligned, draw(aligned, missing, doubtful, p))
  }

  // BGR. Yellow stands out on the red board (a red box did not); cyan marks the doubtful parts.
  private val YELLOW = Scalar(0.0, 230.0, 255.0)
  private val CYAN = Scalar(255.0, 229.0, 0.0)
  private val BLACK = Scalar(0.0, 0.0, 0.0)

  /** The aligned board at half size with a yellow box around every missing part and a cyan one around every doubtful part (black edge for contrast). */
  fun draw(aligned: Mat, missing: List<PartScore>, doubtful: List<PartScore>, p: Params): Mat {
    val scale = 0.5
    val out = Mat()
    Imgproc.resize(aligned, out, Size(), scale, scale, Imgproc.INTER_AREA)
    val pad = p.pxPerMm            // 1 mm around the part box
    val font = Imgproc.FONT_HERSHEY_SIMPLEX
    val placed = ArrayList<IntArray>()   // label plates already drawn (x0, y0, x1, y1): a new label avoids them and stays inside the image
    fun overlaps(a: IntArray, b: IntArray) = a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
    fun mark(s: PartScore, c: Scalar, thick: Int) {
      val b = s.region.bbox
      val x0 = ((b[0] + s.shift[0] - pad) * scale).toInt()
      val y0 = ((b[1] + s.shift[1] - pad) * scale).toInt()
      val x1 = ((b[0] + s.shift[0] + b[2] + pad) * scale).toInt()
      val y1 = ((b[1] + s.shift[1] + b[3] + pad) * scale).toInt()
      val p0 = Point(x0.toDouble(), y0.toDouble())
      val p1 = Point(x1.toDouble(), y1.toDouble())
      Imgproc.rectangle(out, p0, p1, BLACK, thick + 3)      // black edge first, the coloured box on top
      Imgproc.rectangle(out, p0, p1, c, thick)
      val label = s.region.name + " " + fmt("%.2f", s.p)
      val base = IntArray(1)
      val ts = Imgproc.getTextSize(label, font, 0.65, 2, base)
      val tw = ts.width.toInt() + 8
      val th = ts.height.toInt()
      val rightX = max(0, min(x1 - tw, out.cols() - tw - 2))
      val leftX = max(0, min(x0, out.cols() - tw - 2))
      // candidate positions (plate left x, text baseline y): above / below the box, left- or right-aligned, then further above / below
      val cands = listOf(
        intArrayOf(leftX, y0 - 6), intArrayOf(leftX, y1 + th + 6), intArrayOf(rightX, y0 - 6), intArrayOf(rightX, y1 + th + 6),
        intArrayOf(leftX, y0 - th - 14), intArrayOf(rightX, y1 + 2 * th + 14))
      var pick = cands[0]
      for (cd in cands) {
        val r = intArrayOf(cd[0], cd[1] - th - 5, cd[0] + tw, cd[1] + 5)
        if (r[1] >= 0 && r[3] <= out.rows() && placed.none { overlaps(it, r) }) {
          pick = cd
          break
        }
      }
      val plate = intArrayOf(pick[0], pick[1] - th - 5, pick[0] + tw, pick[1] + 5)
      placed.add(plate)
      Imgproc.rectangle(out, Point(plate[0].toDouble(), plate[1].toDouble()), Point(plate[2].toDouble(), plate[3].toDouble()), c, -1)   // yellow (cyan) plate, black text
      Imgproc.putText(out, label, Point(pick[0] + 4.0, pick[1].toDouble()), font, 0.65, BLACK, 2, Imgproc.LINE_AA)
    }
    for (s in doubtful) mark(s, CYAN, 2)
    for (s in missing) mark(s, YELLOW, 4)
    return out
  }
}
