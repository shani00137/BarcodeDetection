using BarcodeDetection.Utilities;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Media;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using ZXingCpp;
using Bitmap = System.Drawing.Bitmap;
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

        public MainWindow()
        {
            InitializeComponent();
            _barcodeReader = new BarcodeReader();
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
                        _cts.Cancel(); // stop video
                        StartCountdown();
                    }
                });

                Cv2.WaitKey(25); // smooth rendering
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

        private string[] TryDecodeBitmap(Bitmap bitmap)
        {
            var detected = new List<string>();

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
    }
}
