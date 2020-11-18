package com.autonomousphonecar;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LaneDetection {

    static int frameCounter = 0;
    static final int NUM_FRAMES = 10;
    static final int POLY_ORDER = 2;
    static Mat betaFramesl = new Mat(POLY_ORDER + 1, 0, CvType.CV_32FC1);
    static Mat betaFramesr = new Mat(POLY_ORDER + 1, 0, CvType.CV_32FC1);

    /**
     * Filter input image for predetecting lanes
     *
     * @param img        input image in RGB format.
     * @param dst        filtered output image
     * @param satThresh  min and max threshold for saturation filtering
     * @param sblxThresh min and max threshold for Sobel filter on the x axis
     */
    public static void pipeline(Mat img, Mat dst, Scalar satThresh, Scalar sblxThresh) {
        // img = undistort(img)
        // img = np.copy(img)
        Mat hls = new Mat();
        List<Mat> hlsChannels = new ArrayList<>(3);
        Mat sobelx = new Mat();
        // Convert to HLS color space and separate the V channel
        Imgproc.cvtColor(img, hls, Imgproc.COLOR_RGB2HLS);
        Core.split(hls, hlsChannels);
        // Sobel x - Take the derivative in x of the light channel
        Imgproc.Sobel(hlsChannels.get(1), sobelx, CvType.CV_8U, 1, 0);
        // Absolute x derivative to accentuate lines away from horizontal
        Core.convertScaleAbs(sobelx, sobelx);
        // Threshold x gradient
        Core.inRange(sobelx, new Scalar(sblxThresh.val[0]), new Scalar(sblxThresh.val[1]), sobelx);
        // Threshold the saturation channel
        Core.inRange(hlsChannels.get(2), new Scalar(satThresh.val[0]), new Scalar(satThresh.val[1]), dst);
        // Combine two thresholds
        Core.bitwise_or(sobelx, dst, dst);

        // memory release
        hls.release();
        sobelx.release();
        for (Mat channel : hlsChannels) {
            channel.release();
        }
    }

    /**
     * find lanes on black and white filtered bird's eye image with histogram and sliding windows
     * method
     *
     * @param img          input image in black and white format.
     * @param outImg       mask of the detected road
     * @param nwindows     number of windows for sliding windows algorithm
     * @param margin       half width of the window
     * @param minpix       minimum number of pixels per window
     * @param draw_windows draw the sliding windows
     */
    public static double calculateLanes(Mat img, Mat outImg, int nwindows, int margin, int minpix, boolean draw_windows) {
        // Make a histogram of each column in the image
        Mat histogram = new Mat();
        Core.reduce(img, histogram, 0, Core.REDUCE_SUM, CvType.CV_32S);
//        colsHist(img.rowRange(img.height() / 2, img.height()), histogram);
//
        // find peaks of left and right halves
        int midpoint = histogram.width() / 2;
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(histogram.colRange(0, midpoint));
        int leftxBase = (int) minMaxLocResult.maxLoc.x;
        minMaxLocResult = Core.minMaxLoc(histogram.colRange(midpoint, histogram.width()));
        int rightxBase = (int) minMaxLocResult.maxLoc.x + midpoint;

        // Set height of windows
        int windowHeight = img.height() / nwindows;
        // Current positions to be updated for each window
        int leftxCurrent = leftxBase;
        int rightxCurrent = rightxBase;

        // Create empty lists to receive left and right lane pixel indices
        MatOfPoint leftLaneInds = new MatOfPoint();
        MatOfPoint rightLaneInds = new MatOfPoint();
        MatOfPoint mop = new MatOfPoint(new Point());

        // Step through the windows one by one
        Mat windows = new Mat(img.size(), CvType.CV_8U, Scalar.all(0));
        for (int window = 0; window < nwindows; window++) {
            // Identify window boundaries in x and y(and right and left)
            int winyLow = img.height() - (window + 1) * windowHeight;
            int winyHigh = img.height() - window * windowHeight;
            int winxLeftLow = leftxCurrent - margin;
            int winxLeftHigh = leftxCurrent + margin;
            int winxRightLow = rightxCurrent - margin;
            int winxRightHigh = rightxCurrent + margin;
            // Keep inside borders
            winyLow = winyLow >= 0 ? winyLow : 0;
            winxLeftLow = winxLeftLow >= 0 ? winxLeftLow : 0;
            winxLeftHigh = winxLeftHigh < midpoint ? winxLeftHigh : midpoint - 1;
            winxRightLow = winxRightLow >= midpoint ? winxRightLow : midpoint;
            winxRightHigh = winxRightHigh < img.width() ? winxRightHigh : img.width() - 1;
            // the windows image
            Mat leftWindow = img.submat(winyLow, winyHigh, winxLeftLow, winxLeftHigh);
            Mat rightWindow = img.submat(winyLow, winyHigh, winxRightLow, winxRightHigh);
            // Draw the windows on the visualization image
            if (draw_windows == true) {
                Imgproc.rectangle(windows, new Point(winxLeftLow, winyLow),
                        new Point(winxLeftHigh, winyHigh), new Scalar(255), 3);
                Imgproc.rectangle(windows, new Point(winxRightLow, winyLow),
                        new Point(winxRightHigh, winyHigh), new Scalar(255), 3);
            }
            MatOfPoint goodLeftInds = new MatOfPoint(), goodRightInds = new MatOfPoint();
            // Identify the nonzero pixels in x and y within the window
            Core.findNonZero(leftWindow, goodLeftInds);
            Core.add(goodLeftInds, new Scalar(winxLeftLow, winyLow), goodLeftInds);
            Core.findNonZero(rightWindow, goodRightInds);
            Core.add(goodRightInds, new Scalar(winxRightLow, winyLow), goodRightInds);
            // Append these indices to the lists
//            leftLaneInds.push_back(goodLeftInds);   // to be deleted
//            rightLaneInds.push_back(goodRightInds); // to be deleted
            double[] means = Core.mean(goodLeftInds).val;
            // If you found > minpix pixels, recenter next window on their mean position
            if (goodLeftInds.height() > minpix) {
                leftxCurrent = (int) means[0];
                mop.setTo(new Scalar(means));
                leftLaneInds.push_back(mop);
            }
            means = Core.mean(goodRightInds).val;
            if (goodRightInds.height() > minpix) {
                rightxCurrent = (int) means[0];
                mop.setTo(new Scalar(means));
                rightLaneInds.push_back(mop);
            }
            // memory release
            leftWindow.release();
            rightWindow.release();
            goodLeftInds.release();
            goodRightInds.release();
        }

        // Fit a second order polynomial to each
        Mat betal = new Mat(POLY_ORDER + 1, 1, CvType.CV_32FC1),
                betar = new Mat(POLY_ORDER + 1, 1, CvType.CV_32FC1);
        if (leftLaneInds.rows() > POLY_ORDER && rightLaneInds.rows() > POLY_ORDER) {

            fitPoly(leftLaneInds, POLY_ORDER, true, betal);
            fitPoly(rightLaneInds, POLY_ORDER, true, betar);

            // calculate polynomial constants with last frames mean
            if (frameCounter < NUM_FRAMES) {
                List<Mat> src = Arrays.asList(betaFramesl, betal);
                Core.hconcat(src, betaFramesl);
                src = Arrays.asList(betaFramesr, betar);
                Core.hconcat(src, betaFramesr);
            } else {
                betal.copyTo(betaFramesl.col(frameCounter % NUM_FRAMES));
                betar.copyTo(betaFramesr.col(frameCounter % NUM_FRAMES));
            }
            frameCounter++;
        }
        if (betaFramesl.cols() > 0 && betaFramesr.cols() > 0) {
            Core.reduce(betaFramesl, betal, 1, Core.REDUCE_AVG);
            Core.reduce(betaFramesr, betar, 1, Core.REDUCE_AVG);
        }


        // calculate steering
        double carPos = img.width() / 2;
        double laneCenterPos = (calculatePoly(img.height() - 1, betal) +
                calculatePoly(img.height() - 1, betar)) / 2;
        double offset = carPos - laneCenterPos;

        double lanesAngle = (calculatePolyAngle(img.height() - 1, betal) +
                calculatePolyAngle(img.height() - 1, betar)) / 2;

        double steeringAngle = lanesAngle + Math.atan(offset / img.height());

        // draw the road in the out image
        Mat road = new Mat(img.size(), CvType.CV_8U, Scalar.all(0));
        fillCurves(betal, betar, road, 8);

        // draw all in out image
        Core.bitwise_or(windows, road, outImg);

        // memory release
        mop.release();
        betal.release();
        betar.release();
        leftLaneInds.release();
        rightLaneInds.release();
        histogram.release();
        road.release();
        windows.release();

        return steeringAngle;
    }

    /**
     * fit a polynomial for a given set of points by regression.
     *
     * @param curve      set of point of the curve to fit
     * @param degree     the degree of the fitted polynomial
     * @param y_function is x is a function of y or the opposite
     * @param beta       output vector of the coefficients of the polynomial
     */
    public static void fitPoly(MatOfPoint curve, int degree, Boolean y_function, Mat beta) {

        int n = curve.rows();

        Mat matrixX = new Mat(n, degree + 1, CvType.CV_32FC1);
        Mat vectorY = new Mat(n, 1, CvType.CV_32FC1);
        Mat vectorX = new Mat(n, 1, CvType.CV_32FC1);

        List<Mat> xys = new ArrayList<>();
        Core.split(curve, xys);

        // build matrices to solve the regression problem
        if (y_function) { // x = f(y)
            xys.get(0).convertTo(vectorY, CvType.CV_32FC1);
            xys.get(1).convertTo(vectorX, CvType.CV_32FC1);
        } else {            // y = f(x)
            xys.get(1).convertTo(vectorY, CvType.CV_32FC1);
            xys.get(0).convertTo(vectorX, CvType.CV_32FC1);
        }
        for (int j = 0; j <= degree; j++)
            Core.pow(vectorX, j, matrixX.col(j));

        Core.solve(matrixX, vectorY, beta, Core.DECOMP_QR);

        // memory release
        vectorX.release();
        vectorY.release();
        matrixX.release();
        xys.get(0).release();
        xys.get(1).release();

    }

    /**
     * Calculate and return the polynomial function on a given point
     *
     * @param x    input point
     * @param beta input vector of the coefficients of the polynomial
     */
    public static double calculatePoly(double x, Mat beta) {
        int n = beta.rows();
        double y = 0;
        for (int i = 0; i < n; i++) {
            y += beta.get(i, 0)[0] * Math.pow(x, i);
        }
        return y;
    }

    /**
     * Fill in the given image the area between two polynomials curves
     *
     * @param betal input vector of the coefficients of the polynomial left curve
     * @param betar input vector of the coefficients of the polynomial right curve
     * @param img   image to draw inside
     * @param step  the distance in pixels between two points of the curves to fill (speed <-> accuracy)
     */
    public static void fillCurves(Mat betal, Mat betar, Mat img, int step) {

        Point[] drawPoints1 = new Point[img.rows() / step + 1];
        Point[] drawPoints2 = new Point[img.rows() / step + 1];
        int n = drawPoints1.length - 1;

        for (int i = 0, ind = 0; ind < n; i += step, ind++) {
            drawPoints1[ind] = new Point(calculatePoly(i, betal), i);
            drawPoints2[ind] = new Point(calculatePoly(i, betar), i);
        }
        // fill bottom of the image
        drawPoints1[n] = new Point(calculatePoly(img.rows() - 1, betal), img.rows() - 1);
        drawPoints2[n] = new Point(calculatePoly(img.rows() - 1, betar), img.rows() - 1);

        // create curves
        MatOfPoint curve1 = new MatOfPoint();
        curve1.fromArray(drawPoints1);

        MatOfPoint curve2 = new MatOfPoint();
        curve2.fromArray(drawPoints2);
        Core.flip(curve2, curve2, 0);

        // create shape and fill it.
        curve2.push_back(curve1);
        List<MatOfPoint> ppt = new ArrayList<>();
        ppt.add(curve2);

        Imgproc.fillPoly(img, ppt, new Scalar(255));

        // memory release
        curve1.release();
        curve2.release();

    }

    /**
     * calculate the slope angle on a given point
     *
     * @param x    input point
     * @param beta input vector of the coefficients of the polynomial function
     */
    public static double calculatePolyAngle(double x, Mat beta) {
        int n = beta.rows();
        double dfdx = 0;
        for (int i = 1; i < n; i++) {
            dfdx += beta.get(i, 0)[0] * i * Math.pow(x, i - 1);
        }
        return Math.atan(dfdx);
    }

}

