using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Yolov5Net.Scorer.Models.Abstract;

namespace Yolov5Net.Scorer.Models
{
    public record YoloNumberPlateModel() : YoloModel(
    640, // Width
    640, // Height
    3,   // Channels
    10,  // 4 + 1 + 5 classes
    new[] { 8, 16, 32 }, // Strides
    new[] // Anchors (use same as your other model unless trained with custom anchors)
    {
        new[] { new[] { 10, 13 }, new[] { 16, 30 }, new[] { 33, 23 } },
        new[] { new[] { 30, 61 }, new[] { 62, 45 }, new[] { 59, 119 } },
        new[] { new[] {116, 90 }, new[] {156,198 }, new[] {373,326 } }
    },
    new[] { 80, 40, 20 }, // Feature map sizes
    0.20f, // Confidence threshold
    0.45f, // IoU threshold
    0.20f, // MulConfidence
    new[] { "output0" }, // ONNX output node name (check with Netron if different)
    new()
    {
        new(0, "LOWERPART_NONOCR"),
        new(1, "LOWERPART_OCR"),
        new(2, "SINGLEPART_OCR"),
        new(3, "UPPERPART_NONOCR"),
        new(4, "UPPERPART_OCR")
    },
    true
);
}
