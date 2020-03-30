package com.auotonomousphonecar;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static android.content.ContentValues.TAG;

public class ImageOperations {

    // fixed size for proposal signs
    public static final int newSizeH = 32, newSizeW = 32;
    public static final String[] shapesNames = {"triangle", "square", "octagon", "circle", "other"};
    // Red for RGB
    public static final Scalar RGB_RED = new Scalar(255, 0, 0);
    // values for colors filter
    private static final Scalar HSV_LOW_RED1 = new Scalar(0, 25, 30);
    private static final Scalar HSV_HIGH_RED1 = new Scalar(5, 250, 200);
    private static final Scalar HSV_LOW_RED2 = new Scalar(150, 25, 30);
    private static final Scalar HSV_HIGH_RED2 = new Scalar(179, 250, 200);
    private static final Scalar HSV_LOW_YELLOW = new Scalar(15, 75, 30);
    private static final Scalar HSV_HIGH_YELLOW = new Scalar(44, 255, 255);
    private static final Scalar HSV_LOW_GREEN = new Scalar(45, 75, 30);
    private static final Scalar HSV_HIGH_GREEN = new Scalar(74, 255, 255);
    private static final Scalar HSV_LOW_BLUE = new Scalar(95, 70, 56);
    private static final Scalar HSV_HIGH_BLUE = new Scalar(130, 250, 128);
    private static final Scalar HSV_LOW_BLACK = new Scalar(0, 0, 0);
    private static final Scalar HSV_HIGH_BLACK = new Scalar(179, 128, 142);
    private static final Scalar HSV_LOW_WHITE = new Scalar(0, 0, 128);
    private static final Scalar HSV_HIGH_WHITE = new Scalar(179, 63, 255);
    private static final Scalar HSV_LOW_ALPHA = new Scalar(128);
    private static final Scalar HSV_HIGH_ALPHA = new Scalar(255);
    // the image changed by findContours
    public static Mat contourImage = new Mat();
    // the found contour as hierarchy vector
    public static Mat hierarchyOutputVector = new Mat();
    // approximated polygonal curve with specified precision
    public static MatOfPoint2f approxCurve = new MatOfPoint2f();
    // the image thresholded for the lower HSV red range
    private static Mat lowerRedRange = new Mat();
    // the image thresholded for the upper HSV red range
    private static Mat upperRedRange = new Mat();

    public static void memoryTest (Mat img){
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2GRAY);
        img.setTo(new Scalar(0,0,0));
        Imgproc.cvtColor(img, upperRedRange, Imgproc.COLOR_RGBA2GRAY);
        Mat a = new Mat();
        Imgproc.cvtColor(img, a, Imgproc.COLOR_RGBA2GRAY);
        a.release();
        Imgproc.cvtColor(upperRedRange, upperRedRange, Imgproc.COLOR_GRAY2RGB);
//        upperRedRange.release();
    }

    /**
     * Filter input image to a given color
     *
     * @param src   input image in HSV format for color, RGBA for alpha, and RGB for achromatic.
     * @param bw    output gray scale image of only '0' and '255' values
     * @param color the wanted color
     */
    public static void colorFilter(Mat src, Mat bw, ColorName color) {

        // threshold the image for the lower and upper HSV red range
        switch (color) {
            case RED:
                Core.inRange(src, HSV_LOW_RED1, HSV_HIGH_RED1, lowerRedRange);
                Core.inRange(src, HSV_LOW_RED2, HSV_HIGH_RED2, upperRedRange);
                // put the two thresholded images together
                Core.addWeighted(lowerRedRange, 1, upperRedRange, 1, 0.0, bw);
                break;
            case BLUE:
                Core.inRange(src, HSV_LOW_BLUE, HSV_HIGH_BLUE, bw);
                break;
            case BLACK:
                Core.inRange(src, HSV_LOW_BLACK, HSV_HIGH_BLACK, bw);
                break;
            case WHITE:
                Core.inRange(src, HSV_LOW_WHITE, HSV_HIGH_WHITE, bw);
                break;
            case YELLOW:
                Core.inRange(src, HSV_LOW_YELLOW, HSV_HIGH_YELLOW, bw);
                break;
            case GREEN:
                Core.inRange(src, HSV_LOW_GREEN, HSV_HIGH_GREEN, bw);
                break;
            case ALPHA:
                Core.inRange(src, HSV_LOW_ALPHA, HSV_HIGH_ALPHA, bw);
                break;
            case RED_ADAPTIVE:
            case BLUE_ADAPTIVE:
                Mat sub1 = new Mat(), sub2 = new Mat();
                MatOfDouble mean = new MatOfDouble(), std = new MatOfDouble();
                List<Mat> rgbchannels = new ArrayList<>();
                // get the rgb channels
                Core.split(src, rgbchannels);
                if(color == ColorName.RED_ADAPTIVE) {
                    Core.subtract(rgbchannels.get(0), rgbchannels.get(1), sub1);
                    Core.subtract(rgbchannels.get(0), rgbchannels.get(2), sub2);
                }
                if(color == ColorName.BLUE_ADAPTIVE) {
                    Core.subtract(rgbchannels.get(2), rgbchannels.get(0), sub1);
                    Core.subtract(rgbchannels.get(2), rgbchannels.get(1), sub2);
                }
                Core.min(sub1, sub2, bw);
                Core.meanStdDev(bw, mean, std);
                Imgproc.threshold(bw, bw, mean.get(0, 0)[0] + 4 * std.get(0, 0)[0],
                        255, CvType.CV_8UC1);
//                Imgproc.adaptiveThreshold(bw, bw, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
//                        Imgproc.THRESH_BINARY, 127, 0);
//                Core.add(rgbchannels.get(0), rgbchannels.get(1), s);
//                Core.add(s, rgbchannels.get(2), s);
                // memory release
                sub1.release();
                sub2.release();
                mean.release();
                std.release();
                for(Mat channel : rgbchannels)
                    channel.release();
                break;
            case ACHROMATIC:
                Mat a = new Mat(), b = new Mat(), c = new Mat();
                List<Mat> channels = new ArrayList<>();
                // get the rgb channels
                Core.split(src, channels);
                Core.absdiff(channels.get(0), channels.get(1), a);
                Core.absdiff(channels.get(0), channels.get(2), b);
                Core.absdiff(channels.get(1), channels.get(2), c);
                Core.add(a, b, bw);
                Core.add(c, bw, bw);
                Core.divide(bw, new Scalar(3 * 20), bw);

                // memory release
                a.release();
                b.release();
                c.release();

        }
        // memory release
//        lowerRedRange.release();
//        upperRedRange.release();
    }

    /**
     * Find triangles and put appropriate labels (in the given image)
     *
     * @param bw          black and white image which the detection is applied on
     * @param dst         output image which the label should apply
     * @param shapes      output list of of points of the detected shapes
     * @param shapesNames input strings with the name of the shape to detect
     */
    public static void shapeDetection(Mat bw, Mat dst, List<Shape> shapes, ColorName color, ShapeName... shapesNames) {

        Rect r;

        boolean triangle = false;
        boolean square = false;
        boolean octagon = false;
        boolean circle = false;

        float scale = dst.width()/(float)bw.width();

        // initialize booleans
        for (ShapeName name : shapesNames) {
            switch (name) {
                case TRIANGLE:
                    triangle = true;
                    break;
                case SQUARE:
                    square = true;
                    break;
                case OCTAGON:
                    octagon = true;
                    break;
                case CIRCLE:
                    circle = true;
                    break;
                case OTHER:
                    triangle = true;
                    square = true;
                    octagon = true;
                    circle = true;
                    break;
            }
        }

        // find contours and store them all as a list
        List<MatOfPoint> contours = new ArrayList<>();
        contourImage = bw.clone();
        Imgproc.findContours(contourImage, contours, hierarchyOutputVector,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // loop over all found contours
        for (MatOfPoint cnt : contours) {

            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

            // approximates a polygonal curve with the specified precision
            Imgproc.approxPolyDP(curve, approxCurve,
                    0.08 * Imgproc.arcLength(curve, true), true);

            int numberVertices = (int) approxCurve.total();
            double contourArea = Imgproc.contourArea(cnt);

            // ignore too small areas
            if (Math.abs(contourArea) < (newSizeH * newSizeW) / 2) {
                continue;
            }

            Log.d(TAG, "vertices:" + numberVertices);

            // triangle detection
            if (numberVertices == 3 && triangle) {

                if (MainActivity.viewMode == MainActivity.VIEW_MODE_SHAPE) {
                    setLabel(dst, "triangle", cnt, scale);
                    continue;
                }

                shapes.add(new Shape(cnt, ShapeName.TRIANGLE, color));
            }


            // many vertices
            else {

                // approximates a polygonal curve with the (higher) specified precision
                Imgproc.approxPolyDP(
                        curve,
                        approxCurve,
                        0.01 * Imgproc.arcLength(curve, true),
                        true
                );


                numberVertices = (int) approxCurve.total();
                contourArea = Imgproc.contourArea(cnt);

                // square detection
                if (numberVertices == 4 && square) {

                    r = Imgproc.boundingRect(cnt);

                    if (Math.abs(1 - ((float) r.width / r.height)) <= 0.25)
                        if (MainActivity.viewMode == MainActivity.VIEW_MODE_SHAPE) {
                            setLabel(dst, "square", cnt, scale);
                            continue;
                        }

                    shapes.add(new Shape(cnt, ShapeName.SQUARE, color));

                }

                // octagon detection
                else if (numberVertices == 8 && octagon) {

                    r = Imgproc.boundingRect(cnt);

                    if (Math.abs(1 - ((float) r.width / r.height)) <= 0.3)
                        if (MainActivity.viewMode == MainActivity.VIEW_MODE_SHAPE) {
                            setLabel(dst, "octagon", cnt, scale);
                            continue;
                        }

                    shapes.add(new Shape(cnt, ShapeName.OCTAGON, color));

                }

                // circle detection
                else if ((numberVertices < 3 || numberVertices > 8) && circle) {

                    r = Imgproc.boundingRect(cnt);
                    int radius = r.width / 2;

                    if (Math.abs(1 - ((float) r.width / r.height)) <= 0.3 &&
                            Math.abs(1 - (contourArea / (Math.PI * radius * radius))) <= 0.15) {

                        if (MainActivity.viewMode == MainActivity.VIEW_MODE_SHAPE) {
                            setLabel(dst, "circle", cnt, scale);
                            continue;
                        }

                        shapes.add(new Shape(cnt, ShapeName.CIRCLE, color));
                    }
                }
            }
        }

    }

    /**
     * Resize a given image to a constant size
     *
     * @param img the image to which to resize
     */
    public static void DownSampleImage(Mat img) {
        Size mSize = new Size(newSizeH, newSizeW);
        Imgproc.resize(img, img, mSize);
    }

    /**
     * Resize a and warp given image to a constant size
     *
     * @param img the image to which to resize
     */
    public static void warpDownSampleImage(Mat img) {
        Scalar sSize = new Scalar(newSizeW, newSizeH);

        MatOfPoint2f mopPoints = new MatOfPoint2f();
        List<Point> points = new ArrayList<>();
        points.add(new Point(0, 0));
        points.add(new Point(1, 0));
        points.add(new Point(0, 1));
        points.add(new Point(1, 1));
        mopPoints.fromList(points);

        perspectiveWarp(img, img, sSize, mopPoints, mopPoints);
    }

    /**
     * Apply Canny filter and dilate
     *
     * @param gray the image to which to find edges
     * @param bw   black and white of images output edges
     */
    public static void applyCanny(Mat gray, Mat bw) {
        // apply canny to get edges only
        Imgproc.Canny(gray, bw, 0, 255);
        // dilate canny output to remove potential holes between edge segments
        Imgproc.dilate(bw, bw, new Mat(), new Point(-1, 1), 1);

    }

    /**
     * Create  cumulative histograms function from image channels
     *
     * @param img      the image to make the LUTs from
     * @param mask     masking te image for histogram
     * @param dstHists destinations Mats for histograms
     */
    public static void createCumulativeHist(Mat img, Mat mask, Mat... dstHists) {
        List<Mat> tmpList = new ArrayList<>();
        tmpList.add(img);
        for (int i = 0; i < dstHists.length; i++) {
            // calculate the histogram
            Imgproc.calcHist(tmpList, new MatOfInt(i), new Mat(), dstHists[i],
                    new MatOfInt(256), new MatOfFloat(0f, 256.0f));
            Mat cumHist = new Mat(dstHists[i].rows(), dstHists[i].cols(), dstHists[i].type());
            // sum to create cumulative histograms
            double sum = 0;
            for (int j = 0; j < dstHists[i].rows(); j++) {
                sum += dstHists[i].get(j, 0)[0];
                cumHist.put(j, 0, sum);
            }
            Core.normalize(cumHist, dstHists[i], 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        }
    }

    /**
     * Create 256 length LUT of the inverse monotone rising function
     *
     * @param srcArray Mat of array with values from 0 to 255 represents the function
     * @param dstLut   int array for lut (size 256)
     */
    public static void createInverseMonoLUT(Mat srcArray, Mat dstLut) {
        byte[] table = new byte[256];
        int j = 0; // index of the source array
        int i = 0; // index of the destination lut
        double last_index, curr_index = 0;
        int last_value, curr_value = 0; // value of the inverse;
        while (i < 256 && j < 256) {
            last_index = curr_index;
            curr_index = srcArray.get(j, 0)[0];
            last_value = curr_value;
            curr_value = j;
            for (; i <= curr_index; i++) {
                // create the LUT
                if (Math.abs(curr_index - i) < Math.abs(i - last_index))
                    table[i] = (byte) curr_value;
                else
                    table[i] = (byte) last_value;
            }
            j++;
        }
        dstLut.put(0, 0, table);
    }

    /**
     * Delete small and asymmetric contours from BW image
     *
     * @param bw input and output image
     */
    public static void deleteSmallContours(Mat bw) {
        Mat stats = new Mat(), labels = new Mat(), mask = new Mat();
        Imgproc.connectedComponentsWithStatsWithAlgorithm(bw, labels, stats, new Mat(), 8,
                CvType.CV_32S, Imgproc.CCL_DEFAULT);
        for (int i = 0; i < stats.rows(); i++) {
            int area[] = new int[1];
            int height[] = new int[1];
            int width[] = new int[1];
            stats.get(i, Imgproc.CC_STAT_AREA, area);
            stats.get(i, Imgproc.CC_STAT_HEIGHT, height);
            stats.get(i, Imgproc.CC_STAT_WIDTH, width);
            if (area[0] < newSizeW * newSizeH ||
                    ((double) height[0] / (double) width[0] > 1.9 ||
                            (double) height[0] / (double) width[0] < 1.0 / 1.9)) {
                Core.compare(labels, new Scalar(i), mask, Core.CMP_EQ);
                bw.setTo(new Scalar(0), mask);
            }
        }
    }

    /**
     * display a label in the center of the given contour (in the given image)
     *
     * @param im      the image to which the label is applied
     * @param label   the label / text to display
     * @param contour the contour to which the label should apply
     */
    public static void setLabel(Mat im, String label, MatOfPoint contour, double scale) {
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
        double fontScale = 1;//0.4;
        int thickness = 2;//1;
        int[] baseline = new int[1];

        Size text = Imgproc.getTextSize(label, fontFace, fontScale, thickness, baseline);
        Rect r = Imgproc.boundingRect(contour);

        Point pt = new Point(
                scale * r.x + ((scale * r.width - text.width) / 2),
                scale * r.y + ((scale * r.height + text.height) / 2)
        );

        Imgproc.putText(im, label, pt, fontFace, scale, RGB_RED, thickness);
    }

    /**
     * Make a 3X3 rotation matrix given angle and axis
     *
     * @param teta the angle (in degrees)
     * @param axis the axis of rotation
     */
    public static Mat setRotationMatrix(float teta, String axis) {

        Mat rotMat = Mat.zeros(3, 3, CvType.CV_32FC1);

        switch (axis) {
            case "x":
                rotMat.put(0, 0, 1);
                rotMat.put(1, 1, Math.cos(teta));
                rotMat.put(2, 1, Math.sin(teta));
                rotMat.put(1, 2, -Math.sin(teta));
                rotMat.put(2, 2, Math.cos(teta));
                break;
            case "y":
                rotMat.put(1, 1, 1);
                rotMat.put(0, 0, Math.cos(teta));
                rotMat.put(0, 2, Math.sin(teta));
                rotMat.put(2, 0, -Math.sin(teta));
                rotMat.put(2, 2, Math.cos(teta));
                break;
            case "z":
                rotMat.put(2, 2, 1);
                rotMat.put(0, 0, Math.cos(teta));
                rotMat.put(1, 0, Math.sin(teta));
                rotMat.put(0, 1, -Math.sin(teta));
                rotMat.put(1, 1, Math.cos(teta));
                break;
        }

        return rotMat;

    }

//    /**
//     * Make a 3X3 rotation matrix given angle and axis
//     *
//     * @param rotationMatrix the angle (in degrees)
//     * returns an array f the yaw, pitch and roll angles.
//     */
//    public static float[] cameraPerspective(float[] rotationMatrix) {
//
//        Mat rotMat = new Mat(3, 3, CvType.CV_32F), F_new, U_new;
//        Scalar F = new Scalar(0, 1, 0), U = new Scalar(0, 0, -1);
//
//        rotMat.put(0, 0, rotationMatrix);
//        Core.mulrotMat.mul()
//
//
//
//
//        return rotMat;
//
//    }

    /**
     * warp image by given transformed dots
     *
     * @param img     the input image
     * @param warped  the output image
     * @param dstSize the output image desired size
     * @param src     4 input source points for making the transform matrix
     * @param dst     4 input transformed points for making the transform matrix
     */
    public static void perspectiveWarp(Mat img, Mat warped, Scalar dstSize, MatOfPoint2f src, MatOfPoint2f dst) {
        Scalar imgSize = new Scalar(img.width(), img.height());
        MatOfPoint2f src2 = new MatOfPoint2f();
        Core.multiply(src, imgSize, src2);
        /*
         * For destination points, I 'm arbitrarily choosing some points to be a nice
         * fit for displaying our warped result again, not exact, but close enough for
         * our purposes
         */
        MatOfPoint2f dst2 = new MatOfPoint2f();
        Core.multiply(dst, dstSize, dst2);
        // Given src and dst points, calculate the perspective transform matrix
        Mat M = Imgproc.getPerspectiveTransform(src2, dst2);
        // Warp the image using OpenCV warpPerspective ()
        Imgproc.warpPerspective(img, warped, M, new Size(dstSize.val[0], dstSize.val[1]));
    }

    /**
     * Make a histogram of each column in the image
     *
     * @param img  the input gray scale image
     * @param hist the output histogram
     */
    public static void colsHist(Mat img, Mat hist) {
        Core.reduce(img, hist, 0, Core.REDUCE_SUM, CvType.CV_32S);
    }


    public enum ShapeName {
        TRIANGLE, SQUARE, OCTAGON, CIRCLE, OTHER
    }

    public enum ColorName {
        RED, BLUE, BLACK, WHITE, YELLOW, GREEN, ALPHA, ACHROMATIC, RED_ADAPTIVE, BLUE_ADAPTIVE, OTHER
    }

    public static class Shape {

        MatOfPoint points;
        ShapeName name;
        ColorName color;

        public Shape(MatOfPoint points_, ShapeName name_, ColorName color_) {
            points = points_;
            name = name_;
            color = color_;
        }
    }


}
