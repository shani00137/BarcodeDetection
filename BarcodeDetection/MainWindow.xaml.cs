using BarcodeDetection.Utilities;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Media;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Yolov5Net.Scorer;
using ZXingCpp;
using Bitmap = System.Drawing.Bitmap;
using Image = SixLabors.ImageSharp.Image;
using PixelFormat = System.Drawing.Imaging.PixelFormat;
using Rectangle = System.Drawing.Rectangle;

namespace BarcodeDetection
{
    public partial class MainWindow : System.Windows.Window
    {
        private CancellationTokenSource _cts;
        private BarcodeReader _barcodeReader;
        private readonly HashSet<string> _detectedBarcodes = new();
        private string _videoPath;
        private DispatcherTimer _countdownTimer;
        private int _countdownSeconds = 10;
        private YoloScorer<YoloBarcodeModel> _scorerBarcodeDetectionModel;

        public MainWindow()
        {
            InitializeComponent();
            _barcodeReader = new BarcodeReader();
            Loaded += MainWindow_LoadedAsync;
            Closed += (_, __) => _scorerBarcodeDetectionModel?.Dispose();
        }

        private void MainWindow_LoadedAsync(object sender, RoutedEventArgs e)
        {
            try
            {
                SessionOptions sessionOptions = new SessionOptions();
                try
                {
                    var options = new SessionOptions();
                    options.EnableMemoryPattern = false; // Can help with compatibility

                    // Enable DirectML with Intel GPU
                    options.AppendExecutionProvider_DML(1); // 0 = default device

                    // Optional: Set graph optimization level
                    options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                    Console.WriteLine("✅ Using DirectML GPU provider");
                }
                catch
                {
                    Console.WriteLine("⚠ Falling back to CPU provider");
                    sessionOptions.AppendExecutionProvider_CPU();
                }


                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Asset", "Weights", "barocode.onnx");
                _scorerBarcodeDetectionModel = new YoloScorer<YoloBarcodeModel>(modelPath, sessionOptions);
           
             
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Initialization failed: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
       
        private void UploadButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Video files|*.mp4;*.avi;*.mkv;*.mov|All files|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                _videoPath = openFileDialog.FileName;
                StartProcessing();
            }
        }

        private void StartProcessing()
        {
            _cts = new CancellationTokenSource();
            ResultsList.ItemsSource = null;
            _detectedBarcodes.Clear();
            _detectedBarcodes.Clear();
            CountdownText.Text = "";
            Task.Run(() => ProcessVideo(_videoPath, _cts.Token));
        }
        #region barcode detection with library
        private void ProcessVideo(string videoPath, CancellationToken token)
        {
            using var capture = new VideoCapture(videoPath);

            if (!capture.IsOpened())
            {
                Dispatcher.Invoke(() => ResultsList.Items.Add("❌ Cannot open video."));
                return;
            }

            using var frame = new Mat();

            while (!token.IsCancellationRequested)
            {
                if (!capture.Read(frame) || frame.Empty())
                    break;

                using var cloned = frame.Clone();
                var bitmap = cloned.ToBitmap();

                // Detect barcodes
                string[] newResults = TryDecodeBitmap(bitmap);

                foreach (var code in newResults)
                {
                    if (_detectedBarcodes.Add(code)) // only new ones
                        MediaBeep.PlayRobotBeep();    // 🔔 beep sound
                }

                // Convert frame to WPF image
                var bitmapImage = cloned.ToBitmapSource();
                bitmapImage.Freeze();

                Dispatcher.Invoke(() =>
                {
                    PreviewImage.Source = bitmapImage;
                    UpdateResultsUI();

                    if (_detectedBarcodes.Count >= 3)
                    {
                        ResultsList.Items.Add("🎉 All 3 barcodes detected!");
                        //_cts.Cancel(); // stop video
                        //StartCountdown();
                    }
                });

                Cv2.WaitKey(1); // smooth rendering
            }
        }

        private void UpdateResultsUI()
        {
            ResultsList.Items.Clear();
            if (_detectedBarcodes.Count > 0)
            {
                int i = 1;
                foreach (var code in _detectedBarcodes)
                {
                    ResultsList.Items.Add($"✅ Barcode {i}: {code}");
                    i++;
                }
            }
            else
            {
                ResultsList.Items.Add("❌ No barcode detected.");
            }
        }

        private void StartCountdown()
        {
            _countdownSeconds = 10;
            CountdownText.Text = _countdownSeconds.ToString();

            _countdownTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            _countdownTimer.Tick += CountdownTick;
            _countdownTimer.Start();
        }

        private void CountdownTick(object sender, EventArgs e)
        {
            _countdownSeconds--;
            CountdownText.Text = _countdownSeconds.ToString();

            if (_countdownSeconds <= 0)
            {
                _countdownTimer.Stop();
                ResultsList.Items.Clear();
                PreviewImage.Source = null;
                StartProcessing(); // restart video processing
            }
        }

        //private string[] TryDecodeBitmap(Bitmap bitmap)
        //{
        //    var detected = new List<string>();

        //    var imgView = BitmapToImageView(bitmap);
        //    var results = BarcodeReader.Read(imgView);

        //    if (results != null && results.Length > 0)
        //        detected.AddRange(results.Select(r => r.Text));

        //    // Try rotations if needed
        //    if (_detectedBarcodes.Count + detected.Count < 3)
        //    {
        //        for (int angle = 90; angle < 360 && _detectedBarcodes.Count < 3; angle += 90)
        //        {
        //            using var rotated = RotateBitmap(bitmap, angle);
        //            var imgViewRotated = BitmapToImageView(rotated);
        //            var rotatedResults = BarcodeReader.Read(imgViewRotated);

        //            if (rotatedResults != null && rotatedResults.Length > 0)
        //            {
        //                foreach (var r in rotatedResults)
        //                {
        //                    if (!_detectedBarcodes.Contains(r.Text))
        //                        detected.Add(r.Text);
        //                }
        //            }
        //        }
        //    }

        //    return detected.ToArray();
        //}
        private string[] TryDecodeBitmap(Bitmap bitmap)
        {
            var detected = new List<string>();

            // --- Clear preview panel for new decode attempt ---
            Dispatcher.Invoke(() => DecodePreviewPanel.Children.Clear());

            //// Show original image
            //ShowDecodePreview(bitmap, "Original");

            var imgView = BitmapToImageView(bitmap);
            var results = BarcodeReader.Read(imgView);

            if (results != null && results.Length > 0)
                detected.AddRange(results.Select(r => r.Text));

            // Try rotations if needed
            if (_detectedBarcodes.Count + detected.Count < 3)
            {
                for (int angle = 90; angle < 360 && _detectedBarcodes.Count < 3; angle += 90)
                {
                    using var rotated = RotateBitmap(bitmap, angle);

                    // --- Show rotated image in UI ---
                    ShowDecodePreview(rotated, $"{angle}°");

                    var imgViewRotated = BitmapToImageView(rotated);
                    var rotatedResults = BarcodeReader.Read(imgViewRotated);

                    if (rotatedResults != null && rotatedResults.Length > 0)
                    {
                        foreach (var r in rotatedResults)
                        {
                            if (!_detectedBarcodes.Contains(r.Text))
                                detected.Add(r.Text);
                        }
                    }
                }
            }

            return detected.ToArray();
        }
        private void ShowDecodePreview(Bitmap bmp, string label)
        {
            var bmpSource = bmp.ToBitmapSource(); // needs OpenCvSharp.WpfExtensions
            bmpSource.Freeze();

            Dispatcher.Invoke(() =>
            {
                var stack = new StackPanel { Orientation = Orientation.Vertical, Margin = new Thickness(5) };

                var imgCtrl = new System.Windows.Controls.Image
                {
                    Source = bmpSource,
                    Width = 120,
                    Margin = new Thickness(2)
                };

                var text = new TextBlock
                {
                    Text = label,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    Foreground = System.Windows.Media.Brushes.White
                };

                stack.Children.Add(imgCtrl);
                stack.Children.Add(text);

                DecodePreviewPanel.Children.Add(stack);
            });
        }



        private Bitmap RotateBitmap(Bitmap bmp, float angle)
        {
            var rotated = new Bitmap(bmp.Width, bmp.Height);
            using var g = Graphics.FromImage(rotated);
            g.TranslateTransform(bmp.Width / 2, bmp.Height / 2);
            g.RotateTransform(angle);
            g.TranslateTransform(-bmp.Width / 2, -bmp.Height / 2);
            g.DrawImage(bmp, new System.Drawing.Point(0, 0));
            return rotated;
        }

        private ImageView BitmapToImageView(Bitmap bitmap)
        {
            var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var bmpData = bitmap.LockBits(rect, ImageLockMode.ReadOnly, bitmap.PixelFormat);

            try
            {
                int bytes = Math.Abs(bmpData.Stride) * bitmap.Height;
                byte[] rgbValues = new byte[bytes];
                Marshal.Copy(bmpData.Scan0, rgbValues, 0, bytes);

                ZXingCpp.ImageFormat format = ZXingCpp.ImageFormat.BGR;
                if (bitmap.PixelFormat == PixelFormat.Format8bppIndexed)
                    format = ZXingCpp.ImageFormat.None;

                return new ImageView(rgbValues, bitmap.Width, bitmap.Height, format, bmpData.Stride);
            }
            finally
            {
                bitmap.UnlockBits(bmpData);
            }
        }
        #endregion

        //#region barcode detection with yolov5
        //// keep global list for crops
        //// Hold 3 candidate crops in memory
        //private readonly List<Bitmap> _candidateBitmaps = new();

        //private void ProcessVideo(string videoPath, CancellationToken token)
        //{
        //    using var capture = new VideoCapture(videoPath);
        //    if (!capture.IsOpened())
        //    {
        //        Dispatcher.Invoke(() => ResultsList.Items.Add("❌ Cannot open video file."));
        //        return;
        //    }

        //    int frameCounter = 0;
        //    int maxFrames = 300; // ~10 seconds at 30fps

        //    using var frame = new Mat();
        //    while (!token.IsCancellationRequested)
        //    {
        //        if (!capture.Read(frame) || frame.Empty())
        //            break;

        //        using var cloned = frame.Clone();
        //        var bitmap = cloned.ToBitmap();

        //        // YOLO detection
        //        using var imageSharp = BitmapToImageSharp(bitmap);
        //        var predictions = _scorerBarcodeDetectionModel.Predict(imageSharp);

        //        foreach (var pred in predictions.OrderByDescending(p => p.Score))
        //        {
        //            if (_candidateBitmaps.Count >= 3)
        //                break;

        //            if (pred.Score < 0.70f)
        //                continue;

        //            int x = (int)pred.Rectangle.X;
        //            int y = (int)pred.Rectangle.Y;
        //            int w = (int)pred.Rectangle.Width;
        //            int h = (int)pred.Rectangle.Height;

        //            Cv2.Rectangle(cloned, new OpenCvSharp.Rect(x, y, w, h), Scalar.Lime, 2);
        //            Cv2.PutText(cloned, $"barcode {pred.Score:P1}",
        //                new OpenCvSharp.Point(x, y - 10),
        //                HersheyFonts.HersheySimplex, 0.7, Scalar.Green, 2);

        //            Rectangle rect = new Rectangle(x, y, w, h);
        //            Bitmap crop = bitmap.Clone(rect, bitmap.PixelFormat);

        //            _candidateBitmaps.Add(crop);

        //            // show preview in sidebar
        //            Dispatcher.Invoke(() =>
        //            {
        //                var stack = new StackPanel { Orientation = Orientation.Vertical, Margin = new Thickness(0, 5, 0, 5) };



        //                stack.Children.Add(new TextBlock
        //                {
        //                    Text = $"⏳ Candidate {_candidateBitmaps.Count} saved ({pred.Score:P0})",
        //                    Foreground = System.Windows.Media.Brushes.White,
        //                    FontSize = 14
        //                });

        //                ResultsList.Items.Add(stack);
        //            });
        //        }

        //        // show live preview always
        //        var bitmapImage = cloned.ToBitmapSource();
        //        bitmapImage.Freeze();
        //        Dispatcher.Invoke(() => PreviewImage.Source = bitmapImage);

        //        frameCounter++;

        //        // If 3 candidates collected, stop and decode
        //        if (_candidateBitmaps.Count >= 3)
        //        {
        //            DecodeCandidates();

        //        }

        //        // If too many frames processed without 3 candidates → reset
        //        if (frameCounter >= maxFrames)
        //        {
        //            frameCounter = 0;
        //            _candidateBitmaps.Clear();
        //            Dispatcher.Invoke(() =>
        //            {
        //                ResultsList.Items.Add("⚠ Timeout reached. Restarting detection...");
        //                ResultsList.Items.Refresh();
        //            });
        //        }

        //        Cv2.WaitKey(25);
        //    }
        //}

        ///// <summary>
        ///// Convert Bitmap → BitmapImage for WPF display
        ///// </summary>
        //private BitmapImage BitmapToBitmapImage(Bitmap bitmap)
        //{
        //    using var memory = new MemoryStream();
        //    bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Png);
        //    memory.Position = 0;

        //    var bitmapImage = new BitmapImage();
        //    bitmapImage.BeginInit();
        //    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
        //    bitmapImage.StreamSource = memory;
        //    bitmapImage.EndInit();
        //    bitmapImage.Freeze();
        //    return bitmapImage;
        //}

        ///// <summary>
        ///// Decode 3 candidate barcodes from memory
        ///// </summary>
        //private void DecodeCandidates()
        //{
        //    Dispatcher.Invoke(() => ResultsList.Items.Add("🔍 Decoding 3 candidates..."));

        //    foreach (var bmp in _candidateBitmaps)
        //    {
        //        var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
        //        var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);

        //        try
        //        {
        //            int channels = System.Drawing.Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
        //            int length = bmpData.Stride * bmp.Height;
        //            byte[] buffer = new byte[length];

        //            Marshal.Copy(bmpData.Scan0, buffer, 0, length);

        //            // ✅ correct constructor order
        //            var format = channels switch
        //            {
        //                4 => ZXingCpp.ImageFormat.RGBA,
        //                3 => ZXingCpp.ImageFormat.RGB,
        //                _ => ZXingCpp.ImageFormat.None
        //            };

        //            var imageView = new ZXingCpp.ImageView(
        //                buffer,
        //                bmp.Width,
        //                bmp.Height,
        //                format,
        //                bmpData.Stride
        //            );

        //            var results = ZXingCpp.BarcodeReader.Read(imageView);

        //            Dispatcher.Invoke(() =>
        //            {
        //                if (results != null && results.Length > 0)
        //                {
        //                    foreach (var r in results)
        //                        ResultsList.Items.Add($"✅ Decoded: {r.Text}");
        //                }
        //                else
        //                {
        //                    ResultsList.Items.Add("❌ Decode failed.");
        //                }
        //            });
        //        }
        //        finally
        //        {
        //            bmp.UnlockBits(bmpData);
        //        }
        //    }

        //    _candidateBitmaps.Clear();
        //}
        //private Image<Rgba32> BitmapToImageSharp(Bitmap bitmap)
        //{
        //    using var ms = new MemoryStream();
        //    bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png); // encode as PNG
        //    ms.Position = 0;
        //    return Image.Load<Rgba32>(ms); // load into ImageSharp
        //}



        //private string[] TryDecodeBitmap(Bitmap bitmap)
        //{
        //    var detected = new List<string>();

        //    // --- Step 1: Add padding around the crop ---
        //    int padding = 20;
        //    var padded = new Bitmap(bitmap.Width + padding * 2, bitmap.Height + padding * 2);
        //    using (var g = Graphics.FromImage(padded))
        //    {
        //        g.Clear(System.Drawing.Color.White); // quiet zone
        //        g.DrawImage(bitmap, new Rectangle(padding, padding, bitmap.Width, bitmap.Height));
        //    }

        //    // --- Step 2: Upscale small crops ---
        //    Bitmap candidate = padded;
        //    if (padded.Width < 200 || padded.Height < 200) // threshold for "too small"
        //    {
        //        candidate = new Bitmap(padded, padded.Width * 2, padded.Height * 2);
        //        padded.Dispose();
        //    }

        //    // --- Step 3: Try decode with options ---
        //    var imgView = BitmapToImageView(candidate);
        //    var options = new ZXingCpp.ReaderOptions
        //    {
        //        TryHarder = true,
        //        TryRotate = true,
        //        TryInvert = true
        //    };
        //    var results = BarcodeReader.Read(imgView, options);

        //    if (results != null && results.Length > 0)
        //        detected.AddRange(results.Select(r => r.Text));

        //    // --- Step 4: Extra fallback with manual rotations ---
        //    if (detected.Count == 0)
        //    {
        //        for (int angle = 90; angle < 360 && detected.Count == 0; angle += 90)
        //        {
        //            using var rotated = RotateBitmap(candidate, angle);
        //            var rotatedView = BitmapToImageView(rotated);
        //            var rotatedResults = BarcodeReader.Read(rotatedView, options);
        //            if (rotatedResults != null && rotatedResults.Length > 0)
        //                detected.AddRange(rotatedResults.Select(r => r.Text));
        //        }
        //    }

        //    candidate.Dispose();
        //    return detected.ToArray();
        //}
        //private Bitmap RotateBitmap(Bitmap bmp, float angle)
        //{
        //    var rotated = new Bitmap(bmp.Width, bmp.Height);
        //    using var g = Graphics.FromImage(rotated);
        //    g.Clear(System.Drawing.Color.White); // background to help decode
        //    g.TranslateTransform(bmp.Width / 2, bmp.Height / 2);
        //    g.RotateTransform(angle);
        //    g.TranslateTransform(-bmp.Width / 2, -bmp.Height / 2);
        //    g.DrawImage(bmp, new System.Drawing.Point(0, 0));
        //    return rotated;
        //}

        //private ImageView BitmapToImageView(Bitmap bitmap)
        //{
        //    var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
        //    var bmpData = bitmap.LockBits(rect, ImageLockMode.ReadOnly, bitmap.PixelFormat);

        //    try
        //    {
        //        int bytes = Math.Abs(bmpData.Stride) * bitmap.Height;
        //        byte[] rgbValues = new byte[bytes];
        //        Marshal.Copy(bmpData.Scan0, rgbValues, 0, bytes);

        //        ZXingCpp.ImageFormat format = ZXingCpp.ImageFormat.BGR;
        //        if (bitmap.PixelFormat == PixelFormat.Format8bppIndexed)
        //            format = ZXingCpp.ImageFormat.BGR;

        //        return new ImageView(rgbValues, bitmap.Width, bitmap.Height, format, bmpData.Stride);
        //    }
        //    finally
        //    {
        //        bitmap.UnlockBits(bmpData);
        //    }
        //}
        //#endregion

    }
}
