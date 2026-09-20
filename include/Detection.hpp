#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Bounding box in pixels: top-left (x, y), width and height.
class Detection
{
public:
    Detection(double x, double y, double width, double height, double confidence, int classId = 0)
        : _x(x), _y(y), _width(width), _height(height), _confidence(confidence), _classId(classId)
    {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) ||
            !std::isfinite(height) || !std::isfinite(confidence) ||
            width <= 0 || height <= 0 || confidence < 0 || confidence > 1)
            throw std::invalid_argument("Invalid detection box or confidence");
    }

    double x() const { return _x; }
    double y() const { return _y; }
    double width() const { return _width; }
    double height() const { return _height; }
    int classId() const { return _classId; }

    double confidence() const { return _confidence; }
    double centerX() const { return _x + _width / 2; }
    double centerY() const { return _y + _height / 2; }

    double iou(const Detection &other) const
    {
        const double w = std::max(0.0, std::min(_x + _width, other._x + other._width) - std::max(_x, other._x));
        const double h = std::max(0.0, std::min(_y + _height, other._y + other._height) - std::max(_y, other._y));
        const double intersection = w * h;
        return intersection / (_width * _height + other._width * other._height - intersection);
    }

private:
    double _x, _y, _width, _height, _confidence;
    int _classId;
};
