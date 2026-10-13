package expo.modules.goldencomparator

import org.json.JSONArray
import org.json.JSONObject
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.features2d.ORB
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import org.opencv.video.DISOpticalFlow
import org.opencv.video.Video
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

/** Photo unusable: board not found / cut off / cannot align / blurred -> RETAKE (never PASS or FAIL). */
class Retake(message: String) : Exception(message)

/** Locale-independent String.format (decimal point even on comma-decimal phones). */
internal fun fmt(format: String, vararg args: Any?): String = String.format(java.util.Locale.US, format, *args)

class Board(
  val canonical: Mat,          // BGR, canonical size (portrait)
  val valid: Mat,              // 0/255: canonical pixels that came from inside the photo
  val hue: Int,
  val fullFrame: Boolean,      // board filled the whole image (pre-cropped, like the golden reference)
  val sharpness: Double,
  val outline: DoubleArray,    // board corners TL, TR, BR, BL as x0,y0,..,x3,y3 in the decoded photo's px
)

class Alignment(
  var h: DoubleArray,          // 3x3 row-major, new canonical -> reference canonical
  val inliers: Int,
  val matches: Int,
  val inlierRatio: Double,
  var residualMm: Double,
  var rotationDeg: Double,
  val method: String,
)

class Region(
  var id: String,
  var name: String,
  var bbox: IntArray,          // x, y, w, h in canonical px
  var kind: String = "",       // "part" | "empty" (footprint / hole that is empty in the reference)
  var partFraction: Double = 0.0,
  var textureStd: Double = 0.0,
)

class Reference(
  val canonical: Mat,
  val valid: Mat,
  val clahe: Mat,              // CLAHE gray (8U)
  val mask: Mat,               // part mask
  val area: Mat,
  val holes: Mat,              // through-hole interiors (don't care)
  val regions: List<Region>,
  val body: Mat,               // body zones of the 'part' regions (255)
  val boardCore: Mat,          // bare board of the reference (255), for the local board colour
  val silver: Mat,             // pads / terminals / solder / silkscreen of the reference (255)
  val structure: Mat,          // structure outside the body zones (255): local alignment is measured on it
  val bodyPx: Map<String, Int>, // region id -> body zone pixels (below body_min_px: not checkable)
)



class InspectionOutput(val result: JSONObject, val aligned: Mat?, val annotated: Mat?)

/**
 * Board detection, rectification and alignment to the golden reference (steps A-C of scripts/golden_compare.py, same steps and same
 * parameters), the reference preparation and the local alignment of the parts. The missing-part decision itself is in Presence.kt.
 * (The old colour / appearance / difference checks that used to live here were removed on 2026-10-12: they did not find really
 * removed parts.) All functions are pure: they take Mats and return new Mats.
 */
object GoldenComparator {
  private val EMPTY_NAME_PREFIXES = listOf("LED", "RED", "SW", "HOLE", "MH", "J", "PIN", "TP")

  // ------------------------------------------------------------------ helpers

  fun ellipse(px: Int): Mat {
    val k = max(1, px).toDouble()
    return Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k, k))
  }

  private fun rect(px: Int): Mat =
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(px.toDouble(), px.toDouble()))

  fun downscale(img: Mat, maxSide: Int): Mat {
    val s = maxSide.toDouble() / max(img.rows(), img.cols())
    if (s >= 1.0) return img
    val out = Mat()
    Imgproc.resize(img, out, Size(), s, s, Imgproc.INTER_AREA)
    return out
  }

  private fun meanStd(m: Mat, mask: Mat? = null): Pair<Double, Double> {
    val mean = MatOfDouble()
    val std = MatOfDouble()
    if (mask == null) Core.meanStdDev(m, mean, std) else Core.meanStdDev(m, mean, std, mask)
    return Pair(mean.toArray()[0], std.toArray()[0])
  }

  /** Dominant saturated hue (0-179) of the central area; circular histogram smoothed +-3 bins. */
  fun dominantHue(hsv: Mat, p: Params, central: Double = 0.5): Int? {
    val h = hsv.rows()
    val w = hsv.cols()
    val y0 = (h * (1 - central) / 2).toInt()
    val y1 = (h * (1 + central) / 2).toInt()
    val x0 = (w * (1 - central) / 2).toInt()
    val x1 = (w * (1 + central) / 2).toInt()
    val c = hsv.submat(y0, y1, x0, x1)
    val sat = Mat()
    Core.inRange(c, Scalar(0.0, p.satMin.toDouble(), p.valMin.toDouble()), Scalar(255.0, 255.0, 255.0), sat)
    if (Core.countNonZero(sat) < 0.10 * c.rows() * c.cols()) return null
    val hist = Mat()
    Imgproc.calcHist(listOf(c), MatOfInt(0), sat, hist, MatOfInt(180), MatOfFloat(0f, 180f))
    val counts = FloatArray(180)
    hist.get(0, 0, counts)
    var best = 0
    var bestVal = -1.0
    for (i in 0 until 180) {
      var s = 0.0
      for (k in -3..3) s += counts[((i - k) % 180 + 180) % 180]
      if (s > bestVal) {
        bestVal = s
        best = i
      }
    }
    return best
  }

  /** Board-colour pixels: circular hue distance <= hue_tol, S >= sat_min, V >= val_min. */
  fun hueMask(hsv: Mat, hue: Int, p: Params): Mat {
    val lo = hue - p.hueTol
    val hi = hue + p.hueTol
    val s = p.satMin.toDouble()
    val v = p.valMin.toDouble()
    val out = Mat()
    fun range(a: Int, b: Int, dst: Mat) =
      Core.inRange(hsv, Scalar(a.toDouble(), s, v), Scalar(b.toDouble(), 255.0, 255.0), dst)
    if (lo >= 0 && hi <= 179) {
      range(lo, hi, out)
    } else {
      val m1 = Mat()
      val m2 = Mat()
      if (lo < 0) {
        range(0, hi, m1)
        range(lo + 180, 179, m2)
      } else {
        range(lo, 179, m1)
        range(0, hi - 180, m2)
      }
      Core.bitwise_or(m1, m2, out)
    }
    return out
  }

  /** TL, TR, BR, BL. */
  fun orderCorners(pts: List<Point>): Array<Point> {
    val s = pts.map { it.x + it.y }
    val d = pts.map { it.y - it.x }
    fun argmin(a: List<Double>) = a.indices.minByOrNull { a[it] }!!
    fun argmax(a: List<Double>) = a.indices.maxByOrNull { a[it] }!!
    return arrayOf(pts[argmin(s)], pts[argmin(d)], pts[argmax(s)], pts[argmax(d)])
  }

  fun claheGray(bgr: Mat): Mat {
    val g = Mat()
    Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY)
    val out = Mat()
    Imgproc.createCLAHE(2.0, Size(8.0, 8.0)).apply(g, out)
    return out
  }

  /** Intersection area / area of the smaller box (boxes x, y, w, h). */
  fun overlap(a: IntArray, b: IntArray): Double {
    val iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    val ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if (iw <= 0 || ih <= 0) return 0.0
    return iw.toDouble() * ih / max(1, min(a[2] * a[3], b[2] * b[3]))
  }

  private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

  // 3x3 matrices, row-major DoubleArray(9)
  private fun mul3(a: DoubleArray, b: DoubleArray): DoubleArray {
    val r = DoubleArray(9)
    for (i in 0 until 3) for (j in 0 until 3) {
      var s = 0.0
      for (k in 0 until 3) s += a[i * 3 + k] * b[k * 3 + j]
      r[i * 3 + j] = s
    }
    return r
  }

  private fun inv3(m: DoubleArray): DoubleArray {
    val (a, b, c) = Triple(m[0], m[1], m[2])
    val (d, e, f) = Triple(m[3], m[4], m[5])
    val (g, h, i) = Triple(m[6], m[7], m[8])
    val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return doubleArrayOf(
      (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
      (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
      (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
    )
  }

  private fun normalized(m: DoubleArray): DoubleArray = DoubleArray(9) { m[it] / m[8] }

  private fun toMat(m: DoubleArray, type: Int = CvType.CV_64F): Mat {
    val out = Mat(3, 3, CvType.CV_64F)
    out.put(0, 0, *m)
    if (type == CvType.CV_64F) return out
    val f = Mat()
    out.convertTo(f, type)
    return f
  }

  private fun fromMat(m: Mat): DoubleArray {
    val d = Mat()
    m.convertTo(d, CvType.CV_64F)
    val out = DoubleArray(9)
    d.get(0, 0, out)
    return out
  }

  private fun applyH(h: DoubleArray, p: Point): Point {
    val w = h[6] * p.x + h[7] * p.y + h[8]
    return Point((h[0] * p.x + h[1] * p.y + h[2]) / w, (h[3] * p.x + h[4] * p.y + h[5]) / w)
  }

  private fun median(values: DoubleArray): Double {
    if (values.isEmpty()) return 0.0
    val s = values.sortedArray()
    val n = s.size
    return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2
  }

  // ------------------------------------------------------------------ A. board detection + rectification

  fun detectBoard(bgr: Mat, p: Params): Board {
    var img = downscale(bgr, p.maxInputSide)
    val h = img.rows()
    val w = img.cols()
    val hsv = Mat()
    Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV)
    val hue = dominantHue(hsv, p) ?: throw Retake("No board found (no board colour in the centre of the image)")
    val mask = hueMask(hsv, hue, p)
    val k = ellipse(max(3, Math.rint(max(h, w) / 300.0).toInt() or 1))
    Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, k)
    Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, k, Point(-1.0, -1.0), 2)
    val contours = ArrayList<MatOfPoint>()
    Imgproc.findContours(mask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
    if (contours.isEmpty()) throw Retake("No board found")
    val contour = contours.maxByOrNull { Imgproc.contourArea(it) }!!
    val filled = Mat.zeros(h, w, CvType.CV_8U)
    Imgproc.drawContours(filled, listOf(contour), -1, Scalar(255.0), Imgproc.FILLED)
    val areaFrac = Imgproc.contourArea(contour) / (h.toDouble() * w)
    if (areaFrac < p.minBoardFraction) throw Retake("No board found (board colour region too small)")

    val m = 2
    val touch = listOf(
      Core.countNonZero(filled.submat(0, m, 0, w)) > 0,
      Core.countNonZero(filled.submat(0, h, w - m, w)) > 0,
      Core.countNonZero(filled.submat(h - m, h, 0, w)) > 0,
      Core.countNonZero(filled.submat(0, h, 0, m)) > 0,
    )
    var corners: Array<Point>
    val fullFrame: Boolean
    if (touch.all { it } && areaFrac > p.fullFrameFraction) {
      // Pre-cropped image (the golden reference): the board fills the frame.
      corners = arrayOf(Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0))
      fullFrame = true
    } else if (touch.any { it }) {
      throw Retake("Board not completely visible (touches the image edge) - retake")
    } else {
      val c2f = MatOfPoint2f(*contour.toArray())
      val approx = MatOfPoint2f()
      Imgproc.approxPolyDP(c2f, approx, 0.02 * Imgproc.arcLength(c2f, true), true)
      val pts = if (approx.rows() == 4) {
        approx.toArray().toList()
      } else {
        val box = arrayOfNulls<Point>(4)
        Imgproc.minAreaRect(c2f).points(box)
        box.map { it!! }
      }
      corners = orderCorners(pts)
      fullFrame = false
    }

    val back = bgr.cols().toDouble() / img.cols()   // downscaled px -> decoded-photo px
    val outline = DoubleArray(8) { (if (it % 2 == 0) corners[it / 2].x else corners[it / 2].y) * back }
    val (tl, tr, br, bl) = corners
    val wpx = (dist(tr, tl) + dist(br, bl)) / 2
    val hpx = (dist(bl, tl) + dist(br, tr)) / 2
    val aspect = max(wpx, hpx) / max(1e-6, min(wpx, hpx))
    val expected = max(p.boardWMm, p.boardHMm) / min(p.boardWMm, p.boardHMm)
    if (abs(aspect / expected - 1) > p.aspectTol) {
      throw Retake(fmt("Board not completely visible (aspect %.2f, expected %.2f) - retake", aspect, expected))
    }
    if (wpx > hpx) corners = arrayOf(tr, br, bl, tl)   // landscape -> portrait; step B resolves 180 deg

    // Area-downscale first when the board is much larger than the canonical frame, then warp linearly.
    val cw = p.canonW
    val ch = p.canonH
    val f = min(1.0, 1.5 * cw / max(1.0, min(wpx, hpx)))
    if (f < 1.0) {
      val small = Mat()
      Imgproc.resize(img, small, Size(), f, f, Imgproc.INTER_AREA)
      img = small
      corners = corners.map { Point(it.x * f, it.y * f) }.toTypedArray()
    }
    val size = Size(cw.toDouble(), ch.toDouble())
    val dst = MatOfPoint2f(Point(0.0, 0.0), Point(cw - 1.0, 0.0), Point(cw - 1.0, ch - 1.0), Point(0.0, ch - 1.0))
    val hm = Imgproc.getPerspectiveTransform(MatOfPoint2f(*corners), dst)
    val canonical = Mat()
    Imgproc.warpPerspective(img, canonical, hm, size, Imgproc.INTER_LINEAR)
    val valid = Mat()
    Imgproc.warpPerspective(Mat(img.rows(), img.cols(), CvType.CV_8U, Scalar(255.0)), valid, hm, size, Imgproc.INTER_NEAREST)
    val gray = Mat()
    Imgproc.cvtColor(canonical, gray, Imgproc.COLOR_BGR2GRAY)
    val lap = Mat()
    Imgproc.Laplacian(gray, lap, CvType.CV_64F)
    val sd = meanStd(lap).second
    val sharp = sd * sd
    if (sharp < p.blurMin) {
      throw Retake(fmt("Image too blurred (sharpness %.0f < %.0f) - retake", sharp, p.blurMin))
    }
    return Board(canonical, valid, hue, fullFrame, sharp, outline)
  }

  // ------------------------------------------------------------------ B. fine alignment

  private class Features(val pts: Array<Point>, val desc: Mat)

  private fun features(gray: Mat, mask: Mat, method: String, n: Int): Features {
    val kps = MatOfKeyPoint()
    val desc = Mat()
    if (method == "SIFT") SIFT.create(n).detectAndCompute(gray, mask, kps, desc)
    else ORB.create(n).detectAndCompute(gray, mask, kps, desc)
    return Features(kps.toArray().map { it.pt }.toTypedArray(), desc)
  }

  /** 3x3 matrix turning canonical points by rot degrees (clockwise) about the canvas centre. */
  private fun rotAboutCentre(rot: Int, w: Int, h: Int): DoubleArray {
    val a = Math.toRadians(rot.toDouble())
    val c = round(cos(a))
    val s = round(sin(a))
    val cx = (w - 1) / 2.0
    val cy = (h - 1) / 2.0
    return doubleArrayOf(c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0, 1.0)
  }

  /**
   * Ratio-test matching restricted to reference keypoints within [radius] px of the expected position.
   * Both images are rectified by their board corners, so after the turn [r] a true match lands within a
   * few mm; restricting the candidates removes the ambiguity of repeated parts. Grid lookup, cell = radius.
   */
  private fun guidedMatches(n: Features, ref: Features, r: DoubleArray, radius: Double, ratio: Double,
                            binary: Boolean): Pair<List<Point>, List<Point>> {
    val src = ArrayList<Point>()
    val dst = ArrayList<Point>()
    if (n.pts.isEmpty() || ref.pts.isEmpty()) return Pair(src, dst)
    val dim = ref.desc.cols()
    val fR: FloatArray?
    val fN: FloatArray?
    val bR: ByteArray?
    val bN: ByteArray?
    if (binary) {
      bR = ByteArray(ref.pts.size * dim).also { ref.desc.get(0, 0, it) }
      bN = ByteArray(n.pts.size * dim).also { n.desc.get(0, 0, it) }
      fR = null
      fN = null
    } else {
      fR = FloatArray(ref.pts.size * dim).also { ref.desc.get(0, 0, it) }
      fN = FloatArray(n.pts.size * dim).also { n.desc.get(0, 0, it) }
      bR = null
      bN = null
    }
    fun distance(i: Int, j: Int): Double {
      if (binary) {
        var s = 0
        for (k in 0 until dim) s += Integer.bitCount((bN!![i * dim + k].toInt() xor bR!![j * dim + k].toInt()) and 0xff)
        return s.toDouble()
      }
      var s = 0.0
      for (k in 0 until dim) {
        val d = (fN!![i * dim + k] - fR!![j * dim + k]).toDouble()
        s += d * d
      }
      return sqrt(s)
    }
    fun key(cx: Int, cy: Int): Long = (cx.toLong() shl 32) or (cy.toLong() and 0xffffffffL)
    val grid = HashMap<Long, MutableList<Int>>()
    for (j in ref.pts.indices) {
      val pt = ref.pts[j]
      grid.getOrPut(key(floor(pt.x / radius).toInt(), floor(pt.y / radius).toInt())) { ArrayList() }.add(j)
    }
    for (i in n.pts.indices) {
      val e = applyH(r, n.pts[i])
      val cx = floor(e.x / radius).toInt()
      val cy = floor(e.y / radius).toInt()
      var best = -1
      var d1 = Double.MAX_VALUE
      var d2 = Double.MAX_VALUE
      var near = 0
      for (gx in cx - 1..cx + 1) for (gy in cy - 1..cy + 1) {
        val cell = grid[key(gx, gy)] ?: continue
        for (j in cell) {
          if (hypot(ref.pts[j].x - e.x, ref.pts[j].y - e.y) > radius) continue
          near++
          val d = distance(i, j)
          if (d < d1) {
            d2 = d1
            d1 = d
            best = j
          } else if (d < d2) {
            d2 = d
          }
        }
      }
      if (best < 0 || (near >= 2 && d1 >= ratio * d2)) continue
      src.add(n.pts[i])
      dst.add(ref.pts[best])
    }
    return Pair(src, dst)
  }

  fun align(new: Board, refCanonical: Mat, refValid: Mat, p: Params): Alignment {
    val rg = claheGray(refCanonical)
    val ng = claheGray(new.canonical)
    val rmask = Mat()
    val nmask = Mat()
    Imgproc.erode(refValid, rmask, ellipse(p.px(1.0)))
    Imgproc.erode(new.valid, nmask, ellipse(p.px(1.0)))
    val cw = p.canonW
    val ch = p.canonH
    var best: Alignment? = null
    var bestPts: Pair<List<Point>, List<Point>>? = null
    for (method in listOf("SIFT", "ORB")) {
      val fr = features(rg, rmask, method, p.features)
      val fn = features(ng, nmask, method, p.features)
      if (fr.pts.size < p.minInliers || fn.pts.size < p.minInliers) continue
      for (rot in intArrayOf(0, 180, 90, 270)) {   // the board may lie turned
        val (src, dst) = guidedMatches(fn, fr, rotAboutCentre(rot, cw, ch), p.px(p.guideRadiusMm).toDouble(),
          p.ratioTest, method == "ORB")
        if (src.size < p.minInliers) continue
        val inl = Mat()
        val hm = Calib3d.findHomography(MatOfPoint2f(*src.toTypedArray()), MatOfPoint2f(*dst.toTypedArray()),
          Calib3d.RANSAC, p.ransacPx, inl)
        if (hm.empty()) continue
        val flags = ByteArray(src.size)
        inl.get(0, 0, flags)
        val nIn = flags.count { it.toInt() != 0 }
        val cand = Alignment(normalized(fromMat(hm)), nIn, src.size, nIn.toDouble() / src.size, 0.0, rot.toDouble(), method)
        if (best == null || cand.inliers > best.inliers) {
          best = cand
          bestPts = Pair(src.filterIndexed { i, _ -> flags[i].toInt() != 0 }, dst.filterIndexed { i, _ -> flags[i].toInt() != 0 })
        }
      }
      val b = best
      if (b != null && b.inliers >= p.minInliers && b.inlierRatio >= p.minInlierRatio) break
    }
    val b = best
    if (b == null || b.inliers < p.minInliers || b.inlierRatio < p.minInlierRatio) {
      val got = if (b != null) fmt("%d inliers, ratio %.2f", b.inliers, b.inlierRatio) else "no matches"
      throw Retake("Cannot align with the golden reference ($got) - retake photo")
    }
    val warpedMask = Mat()
    Imgproc.warpPerspective(nmask, warpedMask, toMat(b.h), Size(cw.toDouble(), ch.toDouble()), Imgproc.INTER_LINEAR)
    val eccMask = Mat()
    Core.bitwise_and(rmask, warpedMask, eccMask)
    b.h = refineEcc(rg, ng, b.h, eccMask, p)
    val (src, dst) = bestPts!!
    b.residualMm = median(DoubleArray(src.size) { dist(applyH(b.h, src[it]), dst[it]) }) / p.pxPerMm
    b.rotationDeg = Math.toDegrees(atan2(b.h[3], b.h[0]))
    val sx = hypot(b.h[0], b.h[3])
    val sy = hypot(b.h[1], b.h[4])
    if (!(p.minScale < sx && sx < p.maxScale && p.minScale < sy && sy < p.maxScale)) {
      throw Retake(fmt("Cannot align (implausible scale %.2f/%.2f) - retake photo", sx, sy))
    }
    return b
  }

  /** Dense refinement of h (new -> ref) by ECC on blurred CLAHE gray, half then full scale. */
  fun refineEcc(refGray: Mat, newGray: Mat, h: DoubleArray, mask: Mat, p: Params): DoubleArray {
    val cw = p.canonW.toDouble()
    val ch = p.canonH.toDouble()
    var w = normalized(inv3(h))                 // ECC warp: ref coords -> new coords
    val crit = TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, p.eccIterations, 1e-5)
    try {
      for (f in doubleArrayOf(0.5, 1.0)) {
        val s = doubleArrayOf(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0)
        val si = inv3(s)
        fun prep(g: Mat): Mat {
          val r = Mat()
          Imgproc.resize(g, r, Size(), f, f, Imgproc.INTER_AREA)
          Imgproc.GaussianBlur(r, r, Size(0.0, 0.0), 1.5)
          val out = Mat()
          r.convertTo(out, CvType.CV_32F)
          return out
        }
        val r = prep(refGray)
        val n = prep(newGray)
        val m = Mat()
        Imgproc.resize(mask, m, Size(r.cols().toDouble(), r.rows().toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        val ws = toMat(mul3(mul3(s, w), si), CvType.CV_32F)
        Video.findTransformECC(r, n, ws, Video.MOTION_HOMOGRAPHY, crit, m, 5)
        w = mul3(mul3(si, fromMat(ws)), s)
      }
    } catch (e: Exception) {
      return h
    }
    val he = normalized(inv3(w))
    val corners = listOf(Point(0.0, 0.0), Point(cw, 0.0), Point(cw, ch), Point(0.0, ch))
    val move = corners.maxOf { dist(applyH(he, it), applyH(h, it)) }
    return if (move / p.pxPerMm <= p.eccMaxMm) he else h
  }

  fun warpToReference(new: Board, al: Alignment, p: Params): Pair<Mat, Mat> {
    val size = Size(p.canonW.toDouble(), p.canonH.toDouble())
    val aligned = Mat()
    val valid = Mat()
    Imgproc.warpPerspective(new.canonical, aligned, toMat(al.h), size, Imgproc.INTER_LINEAR)
    Imgproc.warpPerspective(new.valid, valid, toMat(al.h), size, Imgproc.INTER_NEAREST)
    return Pair(aligned, valid)
  }

  // ------------------------------------------------------------------ C. normalisation + part mask

  /** Board pixels used for comparison: inside the photo, minus a border of border_mm. */
  fun boardArea(valid: Mat, p: Params): Mat {
    val m = Mat.zeros(valid.rows(), valid.cols(), CvType.CV_8U)
    val b = p.px(p.borderMm)
    m.submat(b, valid.rows() - b, b, valid.cols() - b).setTo(Scalar(255.0))
    val out = Mat()
    Core.bitwise_and(m, valid, out)
    return out
  }

  private fun detail(g: Mat, inside: Mat): Double {
    val lap = Mat()
    Imgproc.Laplacian(g, lap, CvType.CV_32F)
    val sd = meanStd(lap, inside).second
    return sd * sd
  }

  private fun blur(src: Mat, sigma: Double): Mat {
    val f = Mat()
    src.convertTo(f, CvType.CV_32F)
    val out = Mat()
    Imgproc.GaussianBlur(f, out, Size(0.0, 0.0), sigma)
    return out
  }

  /**
   * CLAHE gray of both images at EQUAL SHARPNESS, new mean/std matched to the reference.
   * The sharper image (e.g. a contrast-enhanced golden from another camera) is blurred further,
   * sigma in steps of 0.25 up to sharpen_sigma_max, until its detail energy (variance of the Laplacian
   * inside the board) is at most sharpness_ratio x the other one's.
   */
  fun comparisonGrays(refClahe: Mat, newClahe: Mat, area: Mat, p: Params): Triple<Mat, Mat, JSONObject> {
    var r = blur(refClahe, p.sigmaBase)
    var n = blur(newClahe, p.sigmaBase)
    val dr = detail(r, area)
    val dn = detail(n, area)
    val sharperIsRef = dr > dn
    val src = if (sharperIsRef) refClahe else newClahe
    var out = if (sharperIsRef) r else n
    val target = min(dr, dn) * p.sharpnessRatio
    var sig = p.sigmaBase
    while (detail(out, area) > target && sig < p.sharpenSigmaMax) {
      sig += 0.25
      out = blur(src, sig)
    }
    if (sharperIsRef) r = out else n = out
    val (rm, rs) = meanStd(r, area)
    val (nm, ns) = meanStd(n, area)
    val a = rs / max(ns, 1e-6)
    val matched = Mat()
    n.convertTo(matched, CvType.CV_32F, a, rm - nm * a)
    Imgproc.threshold(matched, matched, 255.0, 255.0, Imgproc.THRESH_TRUNC)
    Imgproc.threshold(matched, matched, 0.0, 0.0, Imgproc.THRESH_TOZERO)
    val info = JSONObject()
      .put("sigma_ref", if (sharperIsRef) sig else p.sigmaBase)
      .put("sigma_new", if (sharperIsRef) p.sigmaBase else sig)
      .put("detail_ref", round(dr * 10) / 10)
      .put("detail_new", round(dn * 10) / 10)
    return Triple(r, matched, info)
  }

  /** Pixels that are NOT board colour (bodies, pads, solder, holes); thin silkscreen text removed. */
  fun partMask(bgr: Mat, hue: Int, area: Mat, p: Params): Mat {
    val hsv = Mat()
    Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)
    val m = Mat()
    Core.bitwise_not(hueMask(hsv, hue, p), m)
    Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, ellipse(p.px(p.textOpenMm)))
    val out = Mat()
    Core.bitwise_and(m, area, out)
    return out
  }

  /**
   * Through-holes of the reference: dark, round blobs (hole_min..hole_max mm^2), slightly dilated.
   * A hole shows whatever lies behind the board, so hole interiors are "don't care".
   */
  fun holeMask(refBgr: Mat, area: Mat, p: Params): Mat {
    val hsv = Mat()
    Imgproc.cvtColor(refBgr, hsv, Imgproc.COLOR_BGR2HSV)
    val v = Mat()
    Core.extractChannel(hsv, v, 2)
    val dark = Mat()
    Core.compare(v, Scalar(p.holeValueMax.toDouble()), dark, Core.CMP_LT)
    Core.bitwise_and(dark, area, dark)
    val labels = Mat()
    val stats = Mat()
    val cents = Mat()
    val n = Imgproc.connectedComponentsWithStats(dark, labels, stats, cents, 8, CvType.CV_32S)
    val out = Mat.zeros(dark.rows(), dark.cols(), CvType.CV_8U)
    val amin = p.holeMinMm2 * p.pxPerMm * p.pxPerMm
    val amax = p.holeMaxMm2 * p.pxPerMm * p.pxPerMm
    val st = IntArray(5)
    for (i in 1 until n) {
      stats.get(i, 0, st)
      val (x, y, w, h, a) = st
      if (a < amin || a > amax) continue
      val blob = Mat()
      Core.compare(labels.submat(y, y + h, x, x + w), Scalar(i.toDouble()), blob, Core.CMP_EQ)
      val cnts = ArrayList<MatOfPoint>()
      Imgproc.findContours(blob, cnts, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
      val c = cnts.maxByOrNull { Imgproc.contourArea(it) } ?: continue
      val radius = FloatArray(1)
      Imgproc.minEnclosingCircle(MatOfPoint2f(*c.toArray()), Point(), radius)
      if (a / (PI * max(radius[0].toDouble(), 1e-6) * max(radius[0].toDouble(), 1e-6)) >= p.holeRoundness) {
        out.submat(y, y + h, x, x + w).setTo(Scalar(255.0), blob)
      }
    }
    val dil = Mat()
    Imgproc.dilate(out, dil, ellipse(2 * p.px(p.holeDilateMm) + 1))
    return dil
  }

  fun boardHueOf(canonical: Mat, p: Params): Int {
    val hsv = Mat()
    Imgproc.cvtColor(canonical, hsv, Imgproc.COLOR_BGR2HSV)
    return dominantHue(hsv, p, 0.8) ?: throw Retake("No board colour found after alignment")
  }

  // ------------------------------------------------------------------ D. component regions

  /** LED*, RED*, SW*, HOLE*, MH*, J*, PIN*, TP* and *DNP* footprints are EMPTY at the SMT stage. */
  fun kindFromName(name: String): String {
    val n = name.uppercase()
    return if (EMPTY_NAME_PREFIXES.any { n.startsWith(it) } || n.contains("DNP")) "empty" else "part"
  }

  /** Connected components of the closed reference part mask -> padded, merged boxes (reading order). */
  fun detectRegions(refMask: Mat, p: Params): List<Region> {
    val closed = Mat()
    Imgproc.morphologyEx(refMask, closed, Imgproc.MORPH_CLOSE, ellipse(p.px(p.closeMm)))
    val labels = Mat()
    val stats = Mat()
    val cents = Mat()
    val n = Imgproc.connectedComponentsWithStats(closed, labels, stats, cents, 8, CvType.CV_32S)
    val amin = p.regionMinMm2 * p.pxPerMm * p.pxPerMm
    val amax = p.regionMaxMm2 * p.pxPerMm * p.pxPerMm
    var boxes = ArrayList<IntArray>()
    val st = IntArray(5)
    for (i in 1 until n) {
      stats.get(i, 0, st)
      val (x, y, w, h, a) = st
      if (a < amin) continue
      if (a > amax || (max(w, h) > p.px(p.splitLongMm) && max(w, h).toDouble() / max(1, min(w, h)) > 3)) {
        // very large / long merged blob: split into its separate parts (blobs of the unclosed mask)
        val sub = Mat()
        Core.compare(labels.submat(y, y + h, x, x + w), Scalar(i.toDouble()), sub, Core.CMP_EQ)
        Core.bitwise_and(sub, refMask.submat(y, y + h, x, x + w), sub)
        val sl = Mat()
        val ss = Mat()
        val sc = Mat()
        val sn = Imgproc.connectedComponentsWithStats(sub, sl, ss, sc, 8, CvType.CV_32S)
        val parts = ArrayList<IntArray>()
        for (j in 1 until sn) {
          val s2 = IntArray(5)
          ss.get(j, 0, s2)
          if (s2[4] >= amin && s2[4] <= amax) parts.add(intArrayOf(x + s2[0], y + s2[1], s2[2], s2[3]))
        }
        if (parts.size >= 2) {
          boxes.addAll(parts)
          continue
        }
        if (a > amax) continue
      }
      boxes.add(intArrayOf(x, y, w, h))
    }
    val pad = p.px(p.regionPadMm)
    val hh = refMask.rows()
    val ww = refMask.cols()
    boxes = ArrayList(boxes.map { (x, y, w, h) ->
      val x0 = max(0, x - pad)
      val y0 = max(0, y - pad)
      intArrayOf(x0, y0, min(ww, x + w + pad) - x0, min(hh, y + h + pad) - y0)
    })
    var merged = true
    while (merged) {   // merge boxes overlapping > merge_overlap of the smaller one
      merged = false
      loop@ for (i in boxes.indices) {
        for (j in i + 1 until boxes.size) {
          if (overlap(boxes[i], boxes[j]) > p.mergeOverlap) {
            val a = boxes[i]
            val b = boxes[j]
            val x0 = min(a[0], b[0])
            val y0 = min(a[1], b[1])
            val x1 = max(a[0] + a[2], b[0] + b[2])
            val y1 = max(a[1] + a[3], b[1] + b[3])
            boxes[i] = intArrayOf(x0, y0, x1 - x0, y1 - y0)
            boxes.removeAt(j)
            merged = true
            break@loop
          }
        }
      }
    }
    val row = p.px(3.0)
    val sorted = boxes.sortedWith(compareBy<IntArray>({ (it[1] + it[3] / 2) / row }, { it[0] }))
    return sorted.mapIndexed { k, b ->
      val id = fmt("C%03d", k + 1)
      Region(id, id, b, kindFromName(id))
    }
  }

  /** Reference part-mask fraction and texture std of every region (in place). */
  fun measureRegions(regions: List<Region>, refMask: Mat, refGray: Mat) {
    for (r in regions) {
      val (x, y, w, h) = r.bbox
      if (w <= 0 || h <= 0) continue
      r.partFraction = Core.countNonZero(refMask.submat(y, y + h, x, x + w)).toDouble() / (w * h)
      r.textureStd = meanStd(refGray.submat(y, y + h, x, x + w)).second
      if (r.kind.isEmpty()) r.kind = kindFromName(r.name)
    }
  }

  // ------------------------------------------------------------------ E0. missing-component check

  // A part that was never placed leaves BARE BOARD where the reference shows its body. So every 'part' region has a
  // body zone in the reference (pixels that are neither board colour nor silver pad / solder / glare). In the new photo
  // the share of that zone which shows bare board ("exposure") is small for a present part (even when it sits a little
  // off its position) and large for a missing one. "Bare board" is judged against the board colour measured in a ring
  // around the part, so lighting gradients, white balance and a different camera barely matter.

  /** Local board colour: hue offset from the image's board hue, saturation, brightness (medians). */
  private class Colour(val dh: Double, val s: Double, val v: Double)

  private fun hsvBlur(bgr: Mat, p: Params): Mat {
    val b = Mat()
    Imgproc.GaussianBlur(bgr, b, Size(0.0, 0.0), p.sigmaBase)
    val hsv = Mat()
    Imgproc.cvtColor(b, hsv, Imgproc.COLOR_BGR2HSV)
    return hsv
  }

  /** Signed hue distance to hue0 (-90..89); hue is circular. */
  private fun hueRel(h: Int, hue0: Int): Int = Math.floorMod(h - hue0 + 90, 180) - 90

  /** Median of integer samples given as a histogram (numpy semantics: mean of the two middle samples when even). */
  private fun histMedian(hist: IntArray, offset: Int): Double {
    var n = 0
    for (c in hist) n += c
    if (n == 0) return 0.0
    fun kth(k: Int): Int {
      var acc = 0
      for (i in hist.indices) {
        acc += hist[i]
        if (acc > k) return i + offset
      }
      return hist.size - 1 + offset
    }
    return if (n % 2 == 1) kth(n / 2).toDouble() else (kth(n / 2 - 1) + kth(n / 2)) / 2.0
  }

  /** Median colour of the selected pixels: hsv = interleaved h, s, v bytes; sel = per pixel non-zero byte. */
  private fun medianColour(hsv: ByteArray, sel: ByteArray?, n: Int, hue0: Int): Colour? {
    val hh = IntArray(180)
    val ss = IntArray(256)
    val vv = IntArray(256)
    var count = 0
    for (i in 0 until n) {
      if (sel != null && sel[i].toInt() == 0) continue
      hh[hueRel(hsv[3 * i].toInt() and 0xff, hue0) + 90]++
      ss[hsv[3 * i + 1].toInt() and 0xff]++
      vv[hsv[3 * i + 2].toInt() and 0xff]++
      count++
    }
    if (count == 0) return null
    return Colour(histMedian(hh, -90), histMedian(ss, 0), histMedian(vv, 0))
  }

  private fun bytesOf(m: Mat): ByteArray {
    val c = if (m.isContinuous) m else m.clone()
    val out = ByteArray(c.rows() * c.cols() * c.channels())
    c.get(0, 0, out)
    return out
  }

  private fun globalBoardColour(hsv: Mat, hue0: Int, area: Mat, p: Params): Colour {
    val sel = Mat()
    Core.bitwise_and(hueMask(hsv, hue0, p), area, sel)
    return medianColour(bytesOf(hsv), bytesOf(sel), hsv.rows() * hsv.cols(), hue0) ?: Colour(0.0, 200.0, 200.0)
  }

  /** Box grown by m px, clipped to the image -> x0, y0, x1, y1. */
  private fun boxWindow(box: IntArray, m: Int, rows: Int, cols: Int): IntArray =
    intArrayOf(max(0, box[0] - m), max(0, box[1] - m), min(cols, box[0] + box[2] + m), min(rows, box[1] + box[3] + m))

  private val NO_SHIFT = intArrayOf(0, 0)

  /** Window in reference coordinates, clipped so that the same window moved by `shift` stays inside the image. */
  private fun shiftedWindow(win: IntArray, shift: IntArray, rows: Int, cols: Int): IntArray =
    intArrayOf(max(win[0], -shift[0]), max(win[1], -shift[1]), min(win[2], cols - shift[0]), min(win[3], rows - shift[1]))

  /**
   * Median colour of the bare-board pixels (reference says: board) in a ring around the box. `hsv` is sampled at the
   * reference position + shift (local alignment of the new photo).
   */
  private fun localBoardColour(hsv: Mat, hue0: Int, box: IntArray, core: Mat, fallback: Colour, p: Params,
                               shift: IntArray = NO_SHIFT): Colour {
    val (x0, y0, x1, y1) = shiftedWindow(boxWindow(box, p.px(p.ringOuterMm), hsv.rows(), hsv.cols()), shift,
      hsv.rows(), hsv.cols())
    if (x1 <= x0 || y1 <= y0) return fallback
    val (ix0, iy0, ix1, iy1) = boxWindow(box, p.px(p.ringInnerMm), hsv.rows(), hsv.cols())
    val ww = x1 - x0
    val hh = y1 - y0
    val m = bytesOf(core.submat(y0, y1, x0, x1))
    for (yy in max(iy0, y0) until min(iy1, y1)) for (xx in max(ix0, x0) until min(ix1, x1)) m[(yy - y0) * ww + (xx - x0)] = 0
    var count = 0
    for (b in m) if (b.toInt() != 0) count++
    if (count < p.ringMinPx) return fallback
    val win = hsv.submat(y0 + shift[1], y1 + shift[1], x0 + shift[0], x1 + shift[0])
    return medianColour(bytesOf(win), m, ww * hh, hue0) ?: fallback
  }

  /**
   * (bare board, silver) of an HSV window (interleaved bytes), relative to the local board colour. The reference uses
   * the loose body_* rule (only clearly non-board pixels become body zone), the new photo the tighter board_* rule: a
   * part body that is only a little different from the board (tan capacitors, dark parts in a reddish photo) must still
   * not count as bare board.
   */
  private fun classify(hsv: ByteArray, n: Int, hue0: Int, col: Colour, p: Params, board: BooleanArray, silver: BooleanArray,
                       reference: Boolean = false) {
    val ht = if (reference) p.bodyHueTol else p.boardHueTol
    val sMin = (if (reference) p.bodySK else p.boardSK) * col.s
    val vMin = (if (reference) p.bodyVK else p.boardVK) * col.v
    val sMax = p.silverSK * col.s
    val vSil = p.silverVK * col.v
    for (i in 0 until n) {
      val hd = abs(hueRel(hsv[3 * i].toInt() and 0xff, hue0) - col.dh)
      val s = hsv[3 * i + 1].toInt() and 0xff
      val v = hsv[3 * i + 2].toInt() and 0xff
      val b = hd <= ht && s >= sMin && v >= vMin
      board[i] = b
      silver[i] = !b && s < sMax && v >= vSil
    }
  }

  class RefBody(val body: Mat, val core: Mat, val silver: Mat, val counts: Map<String, Int>)

  /**
   * Body zone of every 'part' region of the reference: pixels that are neither bare board nor silver (relative to the
   * board colour around the part), 1 px eroded; pixels covered by two part boxes belong to no zone. Also the bare-board
   * core and the silver of the whole reference (a removed part leaves its pads and the silkscreen behind).
   */
  fun referenceBody(canonical: Mat, area: Mat, holes: Mat, regions: List<Region>, p: Params): RefBody {
    val hsv = hsvBlur(canonical, p)
    val hue0 = boardHueOf(canonical, p)
    val eroded = Mat()
    Imgproc.erode(hueMask(hsv, hue0, p), eroded, rect(5))
    val core = Mat()
    Core.bitwise_and(eroded, area, core)
    val gcol = globalBoardColour(hsv, hue0, area, p)
    val rows = area.rows()
    val cols = area.cols()
    val n = rows * cols
    val all = bytesOf(hsv)
    val gBoard = BooleanArray(n)
    val gSilver = BooleanArray(n)
    classify(all, n, hue0, gcol, p, gBoard, gSilver, reference = true)
    val silverBytes = ByteArray(n) { if (gSilver[it]) 255.toByte() else 0 }
    val silverAll = Mat(rows, cols, CvType.CV_8U)
    silverAll.put(0, 0, silverBytes)

    val parts = regions.filter { it.kind == "part" }
    val cover = IntArray(n)
    for (r in parts) {
      val (x, y, w, h) = r.bbox
      for (yy in y until y + h) for (xx in x until x + w) cover[yy * cols + xx]++
    }
    val areaB = bytesOf(area)
    val holeB = bytesOf(holes)
    val body = Mat.zeros(rows, cols, CvType.CV_8U)
    val counts = HashMap<String, Int>()
    for (r in parts) {
      val (x, y, w, h) = r.bbox
      val col = localBoardColour(hsv, hue0, r.bbox, core, gcol, p)
      val wn = w * h
      val board = BooleanArray(wn)
      val silver = BooleanArray(wn)
      classify(bytesOf(hsv.submat(y, y + h, x, x + w)), wn, hue0, col, p, board, silver, reference = true)
      val z = ByteArray(wn)
      for (yy in 0 until h) for (xx in 0 until w) {
        val i = yy * w + xx
        val g = (y + yy) * cols + (x + xx)
        // pixels covered by two part boxes are ambiguous: they belong to no body zone
        if (!board[i] && !silver[i] && cover[g] == 1 && areaB[g].toInt() != 0 && holeB[g].toInt() == 0) z[i] = 1
      }
      val zm = Mat(h, w, CvType.CV_8U)
      zm.put(0, 0, z)
      val ze = Mat()
      Imgproc.erode(zm, ze, rect(3))
      val cnt = Core.countNonZero(ze)
      counts[r.id] = cnt
      if (cnt >= p.bodyMinPx) body.submat(y, y + h, x, x + w).setTo(Scalar(255.0), ze)
    }
    return RefBody(body, core, silverAll, counts)
  }


  /**
   * Dense flow (px, CV_32FC2) golden -> new photo of the equal-sharpness comparison grays (DIS optical flow, fast preset).
   * The homography of step B cannot follow a board that is a little bent, or lens distortion: parts of the same photo
   * can sit 1-3 mm away from the reference in one corner and exactly on it in another.
   */
  fun localFlow(refGray: Mat, newGray: Mat): Mat {
    val r8 = Mat()
    val n8 = Mat()
    refGray.convertTo(r8, CvType.CV_8U)   // rounds to nearest and saturates, like np.clip(np.rint(x), 0, 255)
    newGray.convertTo(n8, CvType.CV_8U)
    val flow = Mat()
    DISOpticalFlow.create(DISOpticalFlow.PRESET_FAST).calc(r8, n8, flow)
    return flow
  }

  /**
   * Pixels of the reference with clear structure (pads, traces, silkscreen) that are not part of a body zone: what the
   * local alignment of a part is measured on (a part that is missing must not move its own reference).
   */
  fun referenceStructure(gray: Mat, body: Mat, area: Mat): Mat {
    val gx = Mat()
    val gy = Mat()
    Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0)
    Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1)
    val grad = Mat()
    Core.magnitude(gx, gy, grad)
    val mean = Core.mean(grad, area).`val`[0]
    val above = Mat()
    Core.compare(grad, Scalar(mean), above, Core.CMP_GT)
    val nearBody = Mat()
    Imgproc.dilate(body, nearBody, rect(7))
    val notNear = Mat()
    Core.bitwise_not(nearBody, notNear)
    Core.bitwise_and(above, notNear, above)
    Core.bitwise_and(above, area, above)
    return above
  }

  private fun medianOf(sorted: FloatArray, n: Int): Float =
    if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2f

  /**
   * region id -> (dx, dy) px: where the board around the part lies in the new photo relative to the reference (median
   * flow over the structure pixels in the window around the part). No entry: no local correction (too little structure,
   * or the flow vectors disagree: glare, blur, an obstacle).
   */
  fun regionShifts(flow: Mat, ref: Reference, regions: List<Region>, p: Params): Map<String, IntArray> {
    val out = HashMap<String, IntArray>()
    val rows = flow.rows()
    val cols = flow.cols()
    val fl = FloatArray(rows * cols * 2)
    flow.get(0, 0, fl)
    val st = bytesOf(ref.structure)
    val rad = p.px(p.flowWinMm)
    val lim = p.flowMaxMm * p.pxPerMm
    for (r in regions) {
      if (r.kind != "part") continue
      val (x0, y0, x1, y1) = boxWindow(r.bbox, rad, rows, cols)
      var n = 0
      for (yy in y0 until y1) for (xx in x0 until x1) if (st[yy * cols + xx].toInt() != 0) n++
      if (n < p.flowMinPx) continue
      val fx = FloatArray(n)
      val fy = FloatArray(n)
      var k = 0
      for (yy in y0 until y1) for (xx in x0 until x1) {
        val i = yy * cols + xx
        if (st[i].toInt() != 0) {
          fx[k] = fl[2 * i]
          fy[k] = fl[2 * i + 1]
          k++
        }
      }
      val sx = fx.copyOf().also { it.sort() }
      val sy = fy.copyOf().also { it.sort() }
      val mx = medianOf(sx, n)
      val my = medianOf(sy, n)
      for (i in 0 until n) {
        sx[i] = abs(fx[i] - mx)
        sy[i] = abs(fy[i] - my)
      }
      sx.sort()
      sy.sort()
      if (hypot(medianOf(sx, n).toDouble(), medianOf(sy, n).toDouble()) > p.flowMadMaxMm * p.pxPerMm) continue
      val dx = Math.rint(mx.toDouble()).coerceIn(-lim, lim).toInt()
      val dy = Math.rint(my.toDouble()).coerceIn(-lim, lim).toInt()
      if (dx != 0 || dy != 0) out[r.id] = intArrayOf(dx, dy)
    }
    return out
  }


  internal fun r3(v: Double): Double = round(v * 1000) / 1000

  // ------------------------------------------------------------------ reference

  /** Reference data from the rectified board; regions detected (step D) unless given (e.g. user-edited). */
  fun referenceFromCanonical(canonical: Mat, p: Params, regions: List<Region>?, valid: Mat? = null): Reference {
    val v = valid ?: Mat(canonical.rows(), canonical.cols(), CvType.CV_8U, Scalar(255.0))
    val area = boardArea(v, p)
    val clahe = claheGray(canonical)
    val gray = blur(clahe, p.sigmaBase)
    val mask = partMask(canonical, boardHueOf(canonical, p), area, p)
    val regs = regions ?: detectRegions(mask, p)
    measureRegions(regs, mask, gray)
    val holes = holeMask(canonical, area, p)
    val rb = referenceBody(canonical, area, holes, regs, p)
    return Reference(canonical, v, clahe, mask, area, holes, regs, rb.body, rb.core, rb.silver,
      referenceStructure(gray, rb.body, area), rb.counts)
  }

  fun prepareReference(bgr: Mat, p: Params, regions: List<Region>?): Pair<Board, Reference> {
    val board = detectBoard(bgr, p)
    return Pair(board, referenceFromCanonical(board.canonical, p, regions, board.valid))
  }

  private fun mmList(v: IntArray?, p: Params): Any =
    if (v == null) JSONObject.NULL else JSONArray(v.map { round(it / p.pxPerMm * 100) / 100 })

  private fun centre(v: IntArray?, p: Params): Any =
    if (v == null) JSONObject.NULL
    else JSONArray(listOf(round((v[0] + v[2] / 2.0) / p.pxPerMm * 100) / 100, round((v[1] + v[3] / 2.0) / p.pxPerMm * 100) / 100))


  internal class Prepared(val board: Board, val alignment: Alignment, val aligned: Mat, val valid: Mat,
                         val area: Mat, val coverage: Double, val hue: Int)

  /** Steps A + B for a new photo; throws [Retake] when the photo is unusable. */
  internal fun prepareNew(ref: Reference, newBgr: Mat, p: Params, progress: (String) -> Unit): Prepared {
    val new = detectBoard(newBgr, p)
    progress("Aligning with the golden reference")
    val al = align(new, ref.canonical, ref.valid, p)
    val (aligned, valid) = warpToReference(new, al, p)
    val inBoard = Mat()
    Core.bitwise_and(valid, ref.area, inBoard)
    val cover = Core.countNonZero(inBoard).toDouble() / max(1, Core.countNonZero(ref.area))
    if (cover < p.minCoverage) {
      throw Retake(fmt("Board not completely visible (%.0f%% of the board in the photo) - retake", cover * 100))
    }
    val area = Mat()
    Core.bitwise_and(ref.area, valid, area)
    return Prepared(new, al, aligned, valid, area, cover, boardHueOf(aligned, p))
  }


  // ------------------------------------------------------------------ regions.json

  fun regionsJson(ref: Reference, p: Params): JSONObject {
    val arr = JSONArray()
    for (r in ref.regions) {
      arr.put(JSONObject()
        .put("id", r.id).put("name", r.name).put("kind", r.kind)
        .put("bbox_mm", mmList(r.bbox, p))
        .put("centre_mm", centre(r.bbox, p))
        .put("part_fraction", round(r.partFraction * 10000) / 10000)
        .put("texture_std", round(r.textureStd * 100) / 100)
        .put("body_px", ref.bodyPx[r.id] ?: 0))
    }
    return JSONObject()
      .put("canonical_px", JSONArray(listOf(p.canonW, p.canonH)))
      .put("px_per_mm", p.pxPerMm)
      .put("board_mm", JSONArray(listOf(p.boardWMm, p.boardHMm)))
      .put("regions", arr)
  }

  /** Regions from regions.json (bbox in mm), clamped to the canonical frame. */
  fun parseRegions(json: JSONObject, p: Params): List<Region> {
    val arr = json.getJSONArray("regions")
    val out = ArrayList<Region>()
    for (i in 0 until arr.length()) {
      val o = arr.getJSONObject(i)
      val b = o.getJSONArray("bbox_mm")
      var x = Math.rint(b.getDouble(0) * p.pxPerMm).toInt()
      var y = Math.rint(b.getDouble(1) * p.pxPerMm).toInt()
      var w = Math.rint(b.getDouble(2) * p.pxPerMm).toInt()
      var h = Math.rint(b.getDouble(3) * p.pxPerMm).toInt()
      x = max(0, min(x, p.canonW - 1))
      y = max(0, min(y, p.canonH - 1))
      w = max(1, min(w, p.canonW - x))
      h = max(1, min(h, p.canonH - y))
      val id = o.optString("id", "C%03d".format(i + 1))
      val name = o.optString("name", "").ifEmpty { id }
      out.add(Region(id, name, intArrayOf(x, y, w, h), o.optString("kind", "")))
    }
    return out
  }
}
