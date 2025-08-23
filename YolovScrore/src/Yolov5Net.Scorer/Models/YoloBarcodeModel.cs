using Yolov5Net.Scorer.Models.Abstract;

public record YoloBarcodeModel() : YoloModel(
    640, // Width
    640, // Height
    3,   // Channels
    6,  // 4 + 1 + 5 classes
    new[] { 8, 16, 32 }, // Strides
    new[]
    {
        new[] { new[] { 10, 13 }, new[] { 16, 30 }, new[] { 33, 23 } },
        new[] { new[] { 30, 61 }, new[] { 62, 45 }, new[] { 59, 119 } },
        new[] { new[] {116, 90 }, new[] {156,198 }, new[] {373,326 } }
    },
    new[] { 80, 40, 20 }, // Feature map sizes (P5 model)
    0.20f, // Confidence threshold
    0.45f, // IoU threshold
    0.20f, // MulConfidence
    new[] { "output0" }, // Check your ONNX output node name in Netron
    new()
    {
        new(0, "barcode")
    },
    true
);
