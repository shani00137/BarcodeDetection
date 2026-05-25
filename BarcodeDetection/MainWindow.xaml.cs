using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media.Imaging;
using Yolov5Net.Scorer;
using Dynamsoft.DBR;
using Dynamsoft.CVR;
using Dynamsoft.Core;
using Dynamsoft.License;
using Image = SixLabors.ImageSharp.Image;

namespace BarcodeDetection
{
    public partial class MainWindow : System.Windows.Window
    {
        private YoloScorer<YoloBarcodeModel> _scorerBarcodeDetectionModel;
        private CaptureVisionRouter _cvRouter;

        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
            Closed += (s, e) => { _scorerBarcodeDetectionModel?.Dispose(); _cvRouter?.Dispose(); };
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                // Inside your DecodeCrop method or where you initialize _cvRouter
                string jsonSettings = @"{
    ""CaptureVisionTemplates"": [{
        ""Name"": ""ReadBarcode_Aggressive"",
        ""ImageROIProcessingNameArray"": [""roi-read-barcodes""]
    }],
    ""TargetROIDefOptions"": [{
        ""Name"": ""roi-read-barcodes"",
        ""TaskSettingNameArray"": [""task-read-barcodes""]
    }],
    ""BarcodeReaderTaskOptions"": [{
        ""Name"": ""task-read-barcodes"",
        ""BarcodeFormatIds"": [""BF_ALL""],
        ""ExpectedBarcodesCount"": 1,
        ""LocalizationModes"": [
            { ""Mode"": ""LM_CONNECTED_BLOCKS"" },
            { ""Mode"": ""LM_SCAN_DIRECTLY"" },
            { ""Mode"": ""LM_LINES"" }
        ],
        ""DeblurModes"": [
            { ""Mode"": ""DM_BASED_ON_LOC_BIN"" },
            { ""Mode"": ""DM_THRESHOLD_BINARIZATION"" },
            { ""Mode"": ""DM_GRAY_SCALALR"" }
        ],
        ""ScaleUpModes"": [{ ""Mode"": ""SUM_LINEAR_INTERPOLATION"", ""TargetSize"": 1000 }]
    }]
}";


                // 1. Initialize Dynamsoft License
                string license = "t0082YQEAAFzQsawYFdlbS+MALBl3Cd0W2FyGAEXhHM0cjdJ3LEImqBt//n3FIuogZWd5+KysGEO35u4PfeN8/IsRFy2d1LL+fTPxvhk55BRd2zZJrw==";
                LicenseManager.InitLicense(license, out string errorMsg);
                
                // 2. Initialize YOLO Scorer
                var options = new SessionOptions();
                options.AppendExecutionProvider_CPU(); // Use CPU for single image reliability

                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Asset", "Weights", "barocode.onnx");
                _scorerBarcodeDetectionModel = new YoloScorer<YoloBarcodeModel>(modelPath, options);

                // 3. Initialize Router
                _cvRouter = new CaptureVisionRouter();
                _cvRouter.InitSettings(jsonSettings);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Initialization failed: {ex.Message}");
            }
        }

        private void UploadImageButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Image files|*.jpg;*.jpeg;*.png;*.bmp;*.tiff"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                ProcessStaticImage(openFileDialog.FileName);
            }
        }

        private async Task ProcessStaticImage(string filePath)
        {
            ResultsList.Items.Clear();

            // 1. Load original image
            using var bitmap = new Bitmap(filePath);
            using var mat = bitmap.ToMat();

            // 2. Resize for YOLO (640x640 letterbox)
            using var inputForYolo = ResizeWithLetterbox(mat, 640);
            using var imageSharp = MatToImageSharp(inputForYolo);

            var predictions = _scorerBarcodeDetectionModel.Predict(imageSharp);

            if (!predictions.Any())
            {
                ResultsList.Items.Add("❌ No barcode bounding boxes detected.");
                return;
            }

            // 3. Scale factors (map back to original image)
            float gain = Math.Min(640f / mat.Width, 640f / mat.Height);
            float padX = (640 - mat.Width * gain) / 2;
            float padY = (640 - mat.Height * gain) / 2;

            foreach (var pred in predictions.Where(p => p.Score > 0.4f))
            {
                // Convert back to original coordinates
                int x = (int)((pred.Rectangle.X - padX) / gain);
                int y = (int)((pred.Rectangle.Y - padY) / gain);
                int w = (int)(pred.Rectangle.Width / gain);
                int h = (int)(pred.Rectangle.Height / gain);

                // ✅ Dynamic padding (5% of size OR min 3px)
                int padding = Math.Max(3, (int)(Math.Min(w, h) * 0.10));

                // Expand rectangle
                int newX = Math.Max(0, x - padding);
                int newY = Math.Max(0, y - padding);
                int newW = Math.Min(w + padding * 2, mat.Width - newX);
                int newH = Math.Min(h + padding * 2, mat.Height - newY);

                // Safety check
                if (newW <= 0 || newH <= 0)
                    continue;

                var rect = new System.Drawing.Rectangle(newX, newY, newW, newH);

                // 4. Crop from ORIGINAL image (high quality)
                using var crop = bitmap.Clone(rect, bitmap.PixelFormat);

                // 🔍 Decode
                var decodedText = await ReadBarcodesFromCroppedImage(crop.ToMat());

                
                    ResultsList.Items.Add($"✅ Found: {decodedText}");

                    // Draw rectangle on preview
                    Cv2.Rectangle(
                        mat,
                        new OpenCvSharp.Rect(rect.X, rect.Y, rect.Width, rect.Height),
                        Scalar.Lime,
                        3
                    );
                
            }

            // 5. Show result image
            PreviewImage.Source = mat.ToBitmapSource();
        }

        // Helper to resize while maintaining aspect ratio
        private Mat ResizeWithLetterbox(Mat src, int size)
        {
            float ratio = Math.Min((float)size / src.Width, (float)size / src.Height);
            int newW = (int)(src.Width * ratio);
            int newH = (int)(src.Height * ratio);

            using var resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(newW, newH));

            var dst = new Mat(new OpenCvSharp.Size(size, size), src.Type(), Scalar.Black);
            int top = (size - newH) / 2;
            int left = (size - newW) / 2;

            Cv2.CopyMakeBorder(resized, dst, top, size - newH - top, left, size - newW - left, BorderTypes.Constant, Scalar.Black);
            return dst;
        }

        private Image<Rgba32> MatToImageSharp(Mat mat)
        {
            using var ms = new MemoryStream();
            using var bmp = mat.ToBitmap();
            bmp.Save(ms, ImageFormat.Bmp);
            ms.Position = 0;
            return Image.Load<Rgba32>(ms);
        }
        private string DecodeCrop(Bitmap bmp)
        {
            // 1. Convert to Mat
            using var mat = bmp.ToMat();
            using var processed = new Mat();

            // FIX: Only convert to Gray if it isn't Gray already
            if (mat.Channels() == 3 || mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, processed, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                mat.CopyTo(processed);
            }

            // 2. Add White Padding (Quiet Zone) 
            // Barcode readers fail if bars touch the edge. Let's add 20px of white space.
            using var padded = new Mat();
            Cv2.CopyMakeBorder(processed, padded, 20, 20, 20, 20, BorderTypes.Constant, Scalar.White);

            // 3. Sharpening (Unsharp Mask) to fix the blur in your images
            using var blur = new Mat();
            Cv2.GaussianBlur(padded, blur, new OpenCvSharp.Size(0, 0), 3);
            Cv2.AddWeighted(padded, 1.5, blur, -0.5, 0, padded);

            // 4. Increase Contrast
            Cv2.Normalize(padded, padded, 0, 255, NormTypes.MinMax);

            // 5. Final Step: Decode
            using var finalBmp = padded.ToBitmap();
            BitmapData data = finalBmp.LockBits(new System.Drawing.Rectangle(0, 0, finalBmp.Width, finalBmp.Height),
                ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                int bufferSize = data.Stride * data.Height;
                byte[] buffer = new byte[bufferSize];
                Marshal.Copy(data.Scan0, buffer, 0, bufferSize);

                // Note: Dynamsoft usually expects RGB_888 for 24bpp BitmapData
                ImageData imageData = new ImageData(buffer, finalBmp.Width, finalBmp.Height, data.Stride, EnumImagePixelFormat.IPF_RGB_888);

                CapturedResult capturedResult = _cvRouter.Capture(imageData, "ReadBarcode_Aggressive");

                var barcodeResult = capturedResult?.GetDecodedBarcodesResult();
                return barcodeResult?.GetItems()?.FirstOrDefault()?.GetText() ?? string.Empty;
            }
            catch { return string.Empty; }
            finally { finalBmp.UnlockBits(data); }
        }
        private async Task<List<string>> ReadBarcodesFromCroppedImage(Mat croppedBarcode)
        {
            var results = new HashSet<string>(); // HashSet prevents duplicates

            // Try all 4 rotations — original, 90°, 180°, 270°
            RotateFlags?[] rotations = [null, RotateFlags.Rotate90Clockwise, RotateFlags.Rotate180, RotateFlags.Rotate90Counterclockwise];

            foreach (var rotation in rotations)
            {
                using Mat rotated = new Mat();

                if (rotation.HasValue)
                    Cv2.Rotate(croppedBarcode, rotated, rotation.Value);
                else
                    croppedBarcode.CopyTo(rotated);

                var found = await ScanSingleFrame(rotated);
                foreach (var text in found)
                    results.Add(text);

                // Stop early if we already found all 3 barcodes
                if (results.Count >= 3) break;
            }

            return results.ToList();
        }

        private async Task<List<string>> ScanSingleFrame(Mat frame)
        {
            var list = new List<string>();
            string tempPath = Path.Combine(Path.GetTempPath(), $"barcode_{Guid.NewGuid():N}.png");

            try
            {
                await Task.Run(() => Cv2.ImWrite(tempPath, frame));

                CapturedResult[] results = await Task.Run(() =>
                    _cvRouter!.CaptureMultiPages(tempPath, PresetTemplate.PT_READ_BARCODES));

                foreach (var result in results)
                {
                    var items = result.GetDecodedBarcodesResult()?.GetItems();
                    if (items is null) continue;

                    foreach (var item in items)
                    {
                        string text = item.GetText();
                        if (!string.IsNullOrWhiteSpace(text))
                            list.Add(text);
                    }
                }
            }
            catch (Exception ex)
            {
            }
            finally
            {
                if (File.Exists(tempPath))
                    try { File.Delete(tempPath); } catch { }
            }

            return list;
        }
    }
}