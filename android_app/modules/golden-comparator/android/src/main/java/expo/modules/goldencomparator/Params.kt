package expo.modules.goldencomparator

import org.json.JSONObject
import kotlin.math.max

/**
 * Parameters of the missing-part check. JSON keys (snake_case) are the same as the Params dataclass of the desktop
 * prototype scripts/golden_compare.py; the defaults are the ones the presence model was trained with
 * (ml/presence/predict_photo.py make_params: 20 px/mm, photos not shrunk below 4000 px, min inlier ratio 0.10).
 */
class Params {
  var boardWMm = 83.0          // canonical frame is PORTRAIT (as the golden reference)
  var boardHMm = 116.0
  var pxPerMm = 20.0           // the crops the model sees are cut from the 20 px/mm frame
  var maxInputSide = 4000

  // missing-part decision (model score p(missing) of a part crop)
  var missingThreshold = 0.5   // a part counts as MISSING when p(missing) >= this
  var doubtThreshold = 0.1     // listed as "doubtful" (does not fail the board) between this and the threshold
  var windowCount = 9          // 9 = thorough (every part is looked at through 9 windows, median score), 1 = fast (centre window only)

  // A. board detection + rectification
  var hueTol = 12              // OpenCV hue scale 0-180
  var satMin = 60
  var valMin = 40
  var aspectTol = 0.08
  var minBoardFraction = 0.08
  var fullFrameFraction = 0.80
  var blurMin = 25.0

  // B. fine alignment
  var features = 5000
  var ratioTest = 0.75
  var ransacPx = 3.0
  var minInliers = 30
  var minInlierRatio = 0.10    // 0.25 lost usable photos (150+ inliers) during training
  var guideRadiusMm = 5.0
  var eccIterations = 50
  var eccMaxMm = 3.0
  var minScale = 0.8
  var maxScale = 1.25
  var minCoverage = 0.97

  // C. normalisation of the comparison grays (local alignment is measured on them)
  var sigmaBase = 1.0
  var sharpnessRatio = 1.15
  var sharpenSigmaMax = 4.0
  var textOpenMm = 0.4
  var holeValueMax = 80
  var holeMinMm2 = 0.3
  var holeMaxMm2 = 40.0
  var holeRoundness = 0.72
  var holeDilateMm = 0.2

  // D. component regions of the reference (only used when a reference is prepared)
  var closeMm = 0.3
  var regionMinMm2 = 0.4
  var regionMaxMm2 = 400.0
  var regionPadMm = 0.5
  var mergeOverlap = 0.5
  var splitLongMm = 15.0
  var minTextureStd = 6.0
  var minValidFraction = 0.9

  // reference structure (which pixels the local alignment of a part is measured on) and the local alignment itself
  var bodyHueTol = 12.0
  var bodySK = 0.60
  var bodyVK = 0.60
  var boardHueTol = 6.0
  var boardSK = 0.70
  var boardVK = 0.75
  var silverSK = 0.30
  var silverVK = 0.60
  var flowWinMm = 5.0          // dense optical flow of the board around the part (no body zones)
  var flowMinPx = 150          // needs at least this many structure pixels, else no local correction
  var flowMaxMm = 4.0          // limited to this shift
  var flowMadMaxMm = 0.8       // and must be consistent (median deviation of the flow vectors), else ignored
  var ringInnerMm = 1.0
  var ringOuterMm = 3.0
  var ringMinPx = 200
  var bodyMinPx = 60
  var borderMm = 1.5           // board border excluded from the reference area

  /** mm -> px, rounded half to even like Python's round(). */
  fun px(mm: Double): Int = max(1, Math.rint(mm * pxPerMm).toInt())

  val canonW: Int get() = Math.rint(boardWMm * pxPerMm).toInt()
  val canonH: Int get() = Math.rint(boardHMm * pxPerMm).toInt()

  private val doubles: Map<String, Pair<() -> Double, (Double) -> Unit>> = mapOf(
    "board_w_mm" to Pair({ boardWMm }, { v: Double -> boardWMm = v }),
    "board_h_mm" to Pair({ boardHMm }, { v: Double -> boardHMm = v }),
    "px_per_mm" to Pair({ pxPerMm }, { v: Double -> pxPerMm = v }),
    "missing_threshold" to Pair({ missingThreshold }, { v: Double -> missingThreshold = v }),
    "doubt_threshold" to Pair({ doubtThreshold }, { v: Double -> doubtThreshold = v }),
    "aspect_tol" to Pair({ aspectTol }, { v: Double -> aspectTol = v }),
    "min_board_fraction" to Pair({ minBoardFraction }, { v: Double -> minBoardFraction = v }),
    "full_frame_fraction" to Pair({ fullFrameFraction }, { v: Double -> fullFrameFraction = v }),
    "blur_min" to Pair({ blurMin }, { v: Double -> blurMin = v }),
    "ratio_test" to Pair({ ratioTest }, { v: Double -> ratioTest = v }),
    "ransac_px" to Pair({ ransacPx }, { v: Double -> ransacPx = v }),
    "min_inlier_ratio" to Pair({ minInlierRatio }, { v: Double -> minInlierRatio = v }),
    "guide_radius_mm" to Pair({ guideRadiusMm }, { v: Double -> guideRadiusMm = v }),
    "ecc_max_mm" to Pair({ eccMaxMm }, { v: Double -> eccMaxMm = v }),
    "min_scale" to Pair({ minScale }, { v: Double -> minScale = v }),
    "max_scale" to Pair({ maxScale }, { v: Double -> maxScale = v }),
    "min_coverage" to Pair({ minCoverage }, { v: Double -> minCoverage = v }),
    "sigma_base" to Pair({ sigmaBase }, { v: Double -> sigmaBase = v }),
    "sharpness_ratio" to Pair({ sharpnessRatio }, { v: Double -> sharpnessRatio = v }),
    "sharpen_sigma_max" to Pair({ sharpenSigmaMax }, { v: Double -> sharpenSigmaMax = v }),
    "text_open_mm" to Pair({ textOpenMm }, { v: Double -> textOpenMm = v }),
    "hole_min_mm2" to Pair({ holeMinMm2 }, { v: Double -> holeMinMm2 = v }),
    "hole_max_mm2" to Pair({ holeMaxMm2 }, { v: Double -> holeMaxMm2 = v }),
    "hole_roundness" to Pair({ holeRoundness }, { v: Double -> holeRoundness = v }),
    "hole_dilate_mm" to Pair({ holeDilateMm }, { v: Double -> holeDilateMm = v }),
    "close_mm" to Pair({ closeMm }, { v: Double -> closeMm = v }),
    "region_min_mm2" to Pair({ regionMinMm2 }, { v: Double -> regionMinMm2 = v }),
    "region_max_mm2" to Pair({ regionMaxMm2 }, { v: Double -> regionMaxMm2 = v }),
    "region_pad_mm" to Pair({ regionPadMm }, { v: Double -> regionPadMm = v }),
    "merge_overlap" to Pair({ mergeOverlap }, { v: Double -> mergeOverlap = v }),
    "split_long_mm" to Pair({ splitLongMm }, { v: Double -> splitLongMm = v }),
    "min_texture_std" to Pair({ minTextureStd }, { v: Double -> minTextureStd = v }),
    "min_valid_fraction" to Pair({ minValidFraction }, { v: Double -> minValidFraction = v }),
    "body_hue_tol" to Pair({ bodyHueTol }, { v: Double -> bodyHueTol = v }),
    "body_s_k" to Pair({ bodySK }, { v: Double -> bodySK = v }),
    "body_v_k" to Pair({ bodyVK }, { v: Double -> bodyVK = v }),
    "board_hue_tol" to Pair({ boardHueTol }, { v: Double -> boardHueTol = v }),
    "board_s_k" to Pair({ boardSK }, { v: Double -> boardSK = v }),
    "board_v_k" to Pair({ boardVK }, { v: Double -> boardVK = v }),
    "silver_s_k" to Pair({ silverSK }, { v: Double -> silverSK = v }),
    "silver_v_k" to Pair({ silverVK }, { v: Double -> silverVK = v }),
    "flow_win_mm" to Pair({ flowWinMm }, { v: Double -> flowWinMm = v }),
    "flow_max_mm" to Pair({ flowMaxMm }, { v: Double -> flowMaxMm = v }),
    "flow_mad_max_mm" to Pair({ flowMadMaxMm }, { v: Double -> flowMadMaxMm = v }),
    "ring_inner_mm" to Pair({ ringInnerMm }, { v: Double -> ringInnerMm = v }),
    "ring_outer_mm" to Pair({ ringOuterMm }, { v: Double -> ringOuterMm = v }),
    "border_mm" to Pair({ borderMm }, { v: Double -> borderMm = v }),
  )

  private val ints: Map<String, Pair<() -> Int, (Int) -> Unit>> = mapOf(
    "max_input_side" to Pair({ maxInputSide }, { v: Int -> maxInputSide = v }),
    "window_count" to Pair({ windowCount }, { v: Int -> windowCount = v }),
    "hue_tol" to Pair({ hueTol }, { v: Int -> hueTol = v }),
    "sat_min" to Pair({ satMin }, { v: Int -> satMin = v }),
    "val_min" to Pair({ valMin }, { v: Int -> valMin = v }),
    "features" to Pair({ features }, { v: Int -> features = v }),
    "min_inliers" to Pair({ minInliers }, { v: Int -> minInliers = v }),
    "ecc_iterations" to Pair({ eccIterations }, { v: Int -> eccIterations = v }),
    "hole_value_max" to Pair({ holeValueMax }, { v: Int -> holeValueMax = v }),
    "ring_min_px" to Pair({ ringMinPx }, { v: Int -> ringMinPx = v }),
    "body_min_px" to Pair({ bodyMinPx }, { v: Int -> bodyMinPx = v }),
    "flow_min_px" to Pair({ flowMinPx }, { v: Int -> flowMinPx = v }),
  )

  /** Overrides the parameters present in [json] (unknown keys are ignored). */
  fun load(json: JSONObject): Params {
    for (key in json.keys()) {
      doubles[key]?.second?.invoke(json.getDouble(key))
      ints[key]?.second?.invoke(json.getDouble(key).toInt())
    }
    return this
  }

  fun toJson(): JSONObject {
    val o = JSONObject()
    for ((k, v) in doubles) o.put(k, v.first())
    for ((k, v) in ints) o.put(k, v.first())
    return o
  }

  companion object {
    fun fromJson(text: String?): Params =
      if (text.isNullOrBlank()) Params() else Params().load(JSONObject(text))
  }
}
