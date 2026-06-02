using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models.Local;
using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;

namespace BarcodeDetection
{
    public partial class MainWindow : System.Windows.Window
    {
        private PaddleOcrAll _ocrEngine;

        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
            Closed += (s, e) => _ocrEngine?.Dispose();
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
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
                await ProcessImage(openFileDialog.FileName);
            }
        }

        private async Task ProcessImage(string filePath)
        {
            ResultsList.Items.Clear();

            try
            {
                using var src = Cv2.ImRead(filePath);
                if (src.Empty())
                {
                    ResultsList.Items.Add("Failed to load image.");
                    return;
                }

                PaddleOcrResult result = await Task.Run(() => _ocrEngine.Run(src));

                var regions = result.Regions
                    .Select(r => new { Text = r.Text.Trim(), Center = r.Rect.Center, Size = r.Rect.Size, Angle = r.Rect.Angle })
                    .Where(r => !string.IsNullOrEmpty(r.Text))
                    .ToList();

                if (!regions.Any())
                {
                    ResultsList.Items.Add("No text detected.");
                    return;
                }

                using var displayMat = src.Clone();
                foreach (var region in result.Regions)
                {
                    var rect = region.Rect.BoundingRect();
                    Cv2.Rectangle(displayMat,
                        new OpenCvSharp.Rect(rect.X, rect.Y, rect.Width, rect.Height),
                        Scalar.Lime, 2);
                }

                string ExtractField(string labelPattern, string valuePattern)
                {
                    var labelRegions = regions
                        .Select((r, i) => new { r, i })
                        .Where(x => Regex.IsMatch(x.r.Text, labelPattern, RegexOptions.IgnoreCase))
                        .ToList();

                    foreach (var lr in labelRegions)
                    {
                        string text = lr.r.Text;
                        var match = Regex.Match(text, valuePattern);
                        if (match.Success) return match.Value;

                        var next = regions.Skip(lr.i + 1).FirstOrDefault();
                        if (next != null)
                        {
                            float dy = Math.Abs(next.Center.Y - lr.r.Center.Y);
                            float dx = Math.Abs(next.Center.X - lr.r.Center.X);
                            if (dy < 200 && dx < 800)
                            {
                                match = Regex.Match(next.Text, valuePattern);
                                if (match.Success) return match.Value;
                            }
                        }
                    }

                    var fallback = regions
                        .Select(r => Regex.Match(r.Text, valuePattern))
                        .FirstOrDefault(m => m.Success);
                    return fallback?.Value ?? "Not found";
                }

                string sku = ExtractField(@"sku", @"\d{5,}[A-Za-z][\w-]*");
                string batchNo = ExtractField(@"Batch", @"\d{5,}[-*][\d]+[-*][\w:*<>\-]+");
                string boxNo = "Not found";
                var boxRegions = regions
                    .Select((r, i) => new { r, i })
                    .Where(x => Regex.IsMatch(x.r.Text, @"Box\s*No\.?", RegexOptions.IgnoreCase))
                    .ToList();
                foreach (var lr in boxRegions)
                {
                    var match = Regex.Match(lr.r.Text, @"\d{2,}");
                    if (match.Success) { boxNo = match.Value; break; }
                    var next = regions.Skip(lr.i + 1).FirstOrDefault();
                    if (next != null)
                    {
                        float dy = Math.Abs(next.Center.Y - lr.r.Center.Y);
                        float dx = Math.Abs(next.Center.X - lr.r.Center.X);
                        if (dy < 200 && dx < 800)
                        {
                            match = Regex.Match(next.Text, @"\d{2,}");
                            if (match.Success) { boxNo = match.Value; break; }
                        }
                    }
                }

                ResultsList.Items.Add($"SKU:      {sku}");
                ResultsList.Items.Add($"Batch No: {batchNo}");
                ResultsList.Items.Add($"Box No:   {boxNo}");

                PreviewImage.Source = displayMat.ToBitmapSource();
            }
            catch (Exception ex)
            {
                ResultsList.Items.Add($"Error: {ex.Message}");
            }
        }
    }
}
