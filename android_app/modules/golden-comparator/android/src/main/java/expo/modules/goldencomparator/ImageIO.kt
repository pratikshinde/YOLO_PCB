package expo.modules.goldencomparator

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import kotlin.math.max

/** Image decode / encode helpers. Inputs are never modified: the app copies the original file itself. */
object ImageIO {
  /** "file:///a%20b.jpg" / "/a b.jpg" / "content://..." -> readable stream. */
  private fun open(context: Context, uriOrPath: String): InputStream {
    if (uriOrPath.startsWith("content://")) {
      return context.contentResolver.openInputStream(Uri.parse(uriOrPath))
        ?: throw IllegalArgumentException("Cannot open $uriOrPath")
    }
    return FileInputStream(toPath(uriOrPath))
  }

  fun toPath(uriOrPath: String): String =
    if (uriOrPath.startsWith("file:")) (Uri.parse(uriOrPath).path ?: uriOrPath.removePrefix("file://")) else uriOrPath

  /**
   * Decodes a photo to a BGR Mat with the EXIF orientation applied. Photos with a long side above
   * 4096 px are decoded at a power-of-two sub-sampling (never below 2048 px) to save memory; the
   * pipeline downsizes to 2000 px anyway.
   */
  fun readBgr(context: Context, uriOrPath: String): Mat {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    open(context, uriOrPath).use { BitmapFactory.decodeStream(it, null, bounds) }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IllegalArgumentException("Not an image: $uriOrPath")
    var sample = 1
    val longSide = max(bounds.outWidth, bounds.outHeight)
    if (longSide > 4096) while (longSide / (sample * 2) >= 2048) sample *= 2
    val opts = BitmapFactory.Options().apply {
      inSampleSize = sample
      inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    val bmp = open(context, uriOrPath).use { BitmapFactory.decodeStream(it, null, opts) }
      ?: throw IllegalArgumentException("Cannot decode $uriOrPath")
    val rgba = Mat()
    Utils.bitmapToMat(bmp, rgba)
    bmp.recycle()
    val bgr = Mat()
    Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
    rgba.release()
    val orientation = try {
      open(context, uriOrPath).use {
        ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
      }
    } catch (e: Exception) {
      ExifInterface.ORIENTATION_NORMAL
    }
    return applyOrientation(bgr, orientation)
  }

  fun applyOrientation(m: Mat, orientation: Int): Mat {
    val out = Mat()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> Core.flip(m, out, 1)
      ExifInterface.ORIENTATION_ROTATE_180 -> Core.rotate(m, out, Core.ROTATE_180)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> Core.flip(m, out, 0)
      ExifInterface.ORIENTATION_TRANSPOSE -> Core.transpose(m, out)
      ExifInterface.ORIENTATION_ROTATE_90 -> Core.rotate(m, out, Core.ROTATE_90_CLOCKWISE)
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        Core.transpose(m, out)
        Core.rotate(out, out, Core.ROTATE_180)
      }
      ExifInterface.ORIENTATION_ROTATE_270 -> Core.rotate(m, out, Core.ROTATE_90_COUNTERCLOCKWISE)
      else -> return m
    }
    m.release()
    return out
  }

  fun writePng(uriOrPath: String, m: Mat) {
    val path = toPath(uriOrPath)
    File(path).parentFile?.mkdirs()
    if (!Imgcodecs.imwrite(path, m)) throw IllegalStateException("Cannot write $path")
  }

  fun writeJpg(uriOrPath: String, m: Mat, quality: Int = 92) {
    val path = toPath(uriOrPath)
    File(path).parentFile?.mkdirs()
    if (!Imgcodecs.imwrite(path, m, MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality))) {
      throw IllegalStateException("Cannot write $path")
    }
  }

  fun readBgrFile(uriOrPath: String): Mat {
    val m = Imgcodecs.imread(toPath(uriOrPath), Imgcodecs.IMREAD_COLOR)
    if (m.empty()) throw IllegalArgumentException("Cannot read $uriOrPath")
    return m
  }

  fun readGrayFile(uriOrPath: String): Mat {
    val m = Imgcodecs.imread(toPath(uriOrPath), Imgcodecs.IMREAD_GRAYSCALE)
    if (m.empty()) throw IllegalArgumentException("Cannot read $uriOrPath")
    return m
  }

  fun writeText(uriOrPath: String, text: String) {
    val f = File(toPath(uriOrPath))
    f.parentFile?.mkdirs()
    f.writeText(text, Charsets.UTF_8)
  }

  fun readText(uriOrPath: String): String = File(toPath(uriOrPath)).readText(Charsets.UTF_8)
}
