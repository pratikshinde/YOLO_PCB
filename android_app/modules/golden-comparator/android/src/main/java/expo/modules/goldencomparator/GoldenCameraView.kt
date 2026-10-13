package expo.modules.goldencomparator

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.Observer
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.Promise
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Back camera preview + still capture (CameraX) with tap-to-focus. The still is a 4:3 JPEG
 * (about 12 MP, flash off) with the EXIF rotation set by CameraX.
 */
@SuppressLint("ViewConstructor")
class GoldenCameraView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {
  private val previewView = PreviewView(context).apply {
    // TextureView based preview: unlike the default SurfaceView it is drawn like a normal view, so it cannot turn black
    // inside React Native's clipped / rounded / stacked views.
    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    scaleType = PreviewView.ScaleType.FILL_CENTER
    elevation = 0f
  }
  private val scope = CoroutineScope(
    Dispatchers.Main + SupervisorJob() + CoroutineExceptionHandler { _, t -> Log.e(TAG, "camera coroutine", t) }
  )
  private val onCameraReady by EventDispatcher<Unit>()
  private val onCameraError by EventDispatcher<Map<String, String>>()

  private var camera: Camera? = null
  private var provider: ProcessCameraProvider? = null
  private var imageCapture: ImageCapture? = null
  private var streamObserver: Observer<PreviewView.StreamState>? = null
  private var watchdog: Job? = null

  /** Set from the JS prop; the camera is (re)bound in [recreateCamera] after all props were applied. */
  var active: Boolean = false

  /**
   * CameraX adds its preview surface (TextureView) to [previewView] AFTER the first layout pass and then calls
   * requestLayout(). React Native ignores such requests for native views unless the view opts in, so without this the
   * surface stayed 0x0, never became available and the preview stream stayed IDLE (black picture).
   */
  override val shouldUseAndroidLayout: Boolean = true

  init {
    addView(previewView)
  }

  override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
    val w = MeasureSpec.getSize(widthMeasureSpec)
    val h = MeasureSpec.getSize(heightMeasureSpec)
    previewView.measure(MeasureSpec.makeMeasureSpec(w, MeasureSpec.EXACTLY), MeasureSpec.makeMeasureSpec(h, MeasureSpec.EXACTLY))
    setMeasuredDimension(w, h)
  }

  override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
    // always: this is also what lays out the surface view CameraX adds later
    previewView.layout(0, 0, right - left, bottom - top)
  }

  override fun onViewAdded(child: View?) {
    super.onViewAdded(child)
    if (child == previewView) return
    child?.bringToFront()
    removeView(previewView)
    addView(previewView, 0)
  }

  fun recreateCamera() {
    scope.launch {
      if (!active) {
        unbind()
        return@launch
      }
      if (camera != null) return@launch   // already bound: prop updates must not restart the camera
      try {
        bind()
      } catch (e: Exception) {
        onCameraError(mapOf("message" to "Camera could not be started: ${e.message ?: e.javaClass.simpleName}"))
      }
    }
  }

  @SuppressLint("UnsafeOptInUsageError")
  private suspend fun bind() {
    val cp = ProcessCameraProvider.awaitInstance(context)
    val ratio43 = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
    val previewSelector = ResolutionSelector.Builder().setAspectRatioStrategy(ratio43).build()
    val captureSelector = ResolutionSelector.Builder()
      .setAspectRatioStrategy(ratio43)
      .setResolutionStrategy(ResolutionStrategy(Size(4000, 3000), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
      .build()
    val preview = Preview.Builder().setResolutionSelector(previewSelector).build()
      .also { it.surfaceProvider = previewView.surfaceProvider }
    val capture = ImageCapture.Builder()
      .setResolutionSelector(captureSelector)
      .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
      .setFlashMode(ImageCapture.FLASH_MODE_OFF)
      .build()
    val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(capture).build()
    cp.unbindAll()
    val owner = appContext.throwingActivity as LifecycleOwner
    camera = cp.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, group)
    imageCapture = capture
    provider = cp
    Log.i(TAG, "camera bound, view ${previewView.width}x${previewView.height}, waiting for preview frames")

    // "ready" means: preview frames really arrive (not just: the camera was opened)
    streamObserver?.let { previewView.previewStreamState.removeObserver(it) }
    val observer = Observer<PreviewView.StreamState> { state ->
      Log.i(TAG, "preview stream state: $state")
      if (state == PreviewView.StreamState.STREAMING) {
        watchdog?.cancel()
        onCameraReady(Unit)
      }
    }
    streamObserver = observer
    previewView.previewStreamState.observe(owner, observer)
    watchdog?.cancel()
    watchdog = scope.launch {
      delay(5000)
      val state = previewView.previewStreamState.value
      if (state != PreviewView.StreamState.STREAMING) {
        Log.w(TAG, "no preview frames after 5 s (stream state $state)")
        onCameraError(mapOf("message" to "The camera opened but shows no picture (state: $state). Close other apps that use the camera, then leave this screen and come back."))
      }
    }
  }

  private fun unbind() {
    streamObserver?.let { previewView.previewStreamState.removeObserver(it) }
    streamObserver = null
    watchdog?.cancel()
    provider?.unbindAll()
    camera = null
    imageCapture = null
  }

  /** x, y in dp relative to this view. */
  fun focusAt(xDp: Float, yDp: Float) {
    val cam = camera ?: return
    val d = resources.displayMetrics.density
    val point = previewView.meteringPointFactory.createPoint(xDp * d, yDp * d)
    val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
      .setAutoCancelDuration(4, TimeUnit.SECONDS)
      .build()
    cam.cameraControl.startFocusAndMetering(action)
  }

  fun takePicture(path: String, promise: Promise) {
    val capture = imageCapture ?: return promise.reject("E_NOT_READY", "The camera is not ready yet", null)
    val file = File(ImageIO.toPath(path))
    file.parentFile?.mkdirs()
    capture.takePicture(
      ImageCapture.OutputFileOptions.Builder(file).build(),
      ContextCompat.getMainExecutor(context),
      object : ImageCapture.OnImageSavedCallback {
        override fun onImageSaved(output: ImageCapture.OutputFileResults) {
          promise.resolve(Bundle().apply { putString("uri", Uri.fromFile(file).toString()) })
        }

        override fun onError(exception: ImageCaptureException) {
          promise.reject("E_CAPTURE", exception.message ?: "Capture failed", exception)
        }
      }
    )
  }

  fun cleanup() {
    unbind()
    scope.cancel()
  }

  companion object {
    private const val TAG = "GoldenCameraView"
  }
}
