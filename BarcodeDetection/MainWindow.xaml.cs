using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models.Local;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows;
using Yolov5Net.Scorer;
using Image = SixLabors.ImageSharp.Image;

namespace BarcodeDetection
{
    public partial class MainWindow : System.Windows.Window
    {
        private YoloScorer<YoloBarcodeModel> _scorerBarcodeDetectionModel;
        private PaddleOcrAll _ocrEngine;

        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
            Closed += (s, e) =>
            {
                _scorerBarcodeDetectionModel?.Dispose();
                _ocrEngine?.Dispose();
            };
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                // 1. Initialize YOLO Scorer
                var options = new SessionOptions();
                options.AppendExecutionProvider_CPU();

                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Asset", "Weights", "barocode.onnx");
                _scorerBarcodeDetectionModel = new YoloScorer<YoloBarcodeModel>(modelPath, options);

                // 2. Initialize PaddleOCR
                _ocrEngine = new PaddleOcrAll(LocalFullModels.EnglishV4, PaddleDevice.Mkldnn())
                {
                    AllowRotateDetection = true,
                };
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Initialization failed: {ex.Message}");
            }
        }

        private async void UploadImageButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Image files|*.jpg;*.jpeg;*.png;*.bmp;*.tiff"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                await ProcessStaticImage(openFileDialog.FileName);
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
                ResultsList.Items.Add("No text regions detected.");
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

                // Dynamic padding (5% of size OR min 3px)
                int padding = Math.Max(3, (int)(Math.Min(w, h) * 0.10));

                // Expand rectangle
                int newX = Math.Max(0, x - padding);
                int newY = Math.Max(0, y - padding);
                int newW = Math.Min(w + padding * 2, mat.Width - newX);
                int newH = Math.Min(h + padding * 2, mat.Height - newY);

                if (newW <= 0 || newH <= 0)
                    continue;

                var rect = new System.Drawing.Rectangle(newX, newY, newW, newH);

                // 4. Crop from ORIGINAL image (high quality)
                using var crop = bitmap.Clone(rect, bitmap.PixelFormat);

                // 5. Run OCR on cropped image
                var resultText = await Task.Run(() =>
                {
                    using var cropMat = crop.ToMat();
                    PaddleOcrResult result = _ocrEngine.Run(cropMat);
                    return result.Text?.Trim() ?? string.Empty;
                });

                if (!string.IsNullOrWhiteSpace(resultText))
                {
                    ResultsList.Items.Add(resultText);
                }

                // Draw rectangle on preview
                Cv2.Rectangle(
                    mat,
                    new OpenCvSharp.Rect(rect.X, rect.Y, rect.Width, rect.Height),
                    Scalar.Lime,
                    3
                );
            }

            // 6. Show result image
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
    }
}
