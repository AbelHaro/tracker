#pragma once

// Pixel coordinates accepted by the NumPy input adapter.
enum class BoxFormat
{
    TLWH,   // Top-left x, y, width, height.
    CXCYWH, // Center x, y, width, height (YOLO).
    XYXY    // Top-left x, y, bottom-right x, y.
};
