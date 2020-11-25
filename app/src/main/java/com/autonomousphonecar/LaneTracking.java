package com.autonomousphonecar;

//import org.apache.commons.math3.fitting.PolynomialCurveFitter;
//import org.apache.commons.math3.fitting.WeightedObservedPoints;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class LaneTracking {

    static final int NUM_FRAMES = 10;
    static final int POLY_ORDER = 3;
    static final float ANGLE_OFFSET = 0.0f;

    private float mLanesWidth; // lane width in millimeters

    private float mFocalLength; // focal length of the camera
    private float mSensorWidth; // width of camera sensor in millimeters
    private float mSensorHeight; // height of camera sensor in millimeters
    private float mSensorHorizontalAngle; // horizontal angle of camera
    private float mSensorVerticalAngle;  // vertical angle of camera
    private Mat mCameraMatrix = new Mat(3, 3, CvType.CV_64FC1);

    private Size mLanesProjectedSize;
    private float mLanesFocalLength; // focal length of the camera
    private float mLanesSensorWidth; // width of camera sensor in millimeters
    private float mLanesSensorHeight; // height of camera sensor in millimeters
    private float mLanesSensorHorizontalAngle; // horizontal angle of transformed lane camera
    private float mLanesSensorVerticalAngle;  // vertical angle of transformed lane camera
    private Mat mLanesCameraMatrix = new Mat(3, 3, CvType.CV_64FC1);

    public float mPitchAngle;
    public float mRollAngle;

    public Direction mNextTurn = Direction.STRAIGHT;
    double[] mLeftLaneDir = new double[2];
    double[] mRightLaneDir = new double[2];
    public int mSpeedLimit = 150; // between 0 to 150
    public int mPower = 100;  // between 100 to -100
    public int mSteering = 0; // between 100 to -100
    public double mOffset = 0; // in pixels
    public double mCurvatureRadius; // in pixels

    private Mat mPolyBetasL = new Mat(POLY_ORDER + 1, 0, CvType.CV_64FC1);
    private Mat mPolyBetasR = new Mat(POLY_ORDER + 1, 0, CvType.CV_64FC1);
    private Mat mLinearBetasL = new Mat(2, 0, CvType.CV_64FC1);
    private Mat mLinearBetasR = new Mat(2, 0, CvType.CV_64FC1);
    private boolean mIsJunction = false;
    private int mPolyCounter = 0;
    private int mLinearCounter = 0;
    private int mJunctionCounter = 0;
    private double mRightAngle = 0;
    private double mLeftAngle = 0;

    private Mat mMask = new Mat();
    private Mat mIntermediate = new Mat();
    private Mat mM = new Mat(3, 3, CvType.CV_64FC1);

    /**
     * Calculate inside class lanes variables and return warped lanes image
     *
     * @param src     input image
     * @param roadImg warped road image
     */
    public void LaneTrackingPipeline(Mat src, Mat roadImg) {

        birdsEyeTransform(src, mIntermediate, mM);

//        // debugging
//        filterPipeline(mIntermediate, roadImg);
//        return;

        filterPipeline(mIntermediate, mMask);

        Mat grayScale = Mat.zeros(mMask.size(), CvType.CV_8UC1);
        lanesDetection(mMask, grayScale);

        roadImg.create(grayScale.size(), grayScale.type());
        grayScale.copyTo(roadImg);

        // memory release
        grayScale.release();

    }


    /**
     * Draw the lanes image unwarped in the out image and pring power and steeting
     *
     * @param src     input image that the lanes will draw on
     * @param roadImg warped grayscale lanes image
     * @param outImg  output image
     */
    public void drawLanes(Mat src, Mat roadImg, Mat outImg) {

        Mat warpImg = new Mat();

        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
        float scale = 2;
        int thickness = 2;

        Imgproc.warpPerspective(roadImg, warpImg, mM.inv(), src.size());

        outImg.create(src.size(), src.type());
        src.copyTo(outImg);
        Core.add(src, new Scalar(0, 32, 64), outImg, warpImg);

        String label = String.format("steering: %d", mSteering);
        Point pt = new Point(100, 300);
        Imgproc.putText(outImg, label, pt, fontFace, scale, new Scalar(200, 200, 255), thickness);

        label = String.format("power: %d", mPower);
        pt = new Point(100, 400);
        Imgproc.putText(outImg, label, pt, fontFace, scale, new Scalar(200, 200, 255), thickness);

        scale = 1;
        thickness = 1;

        label = String.format("offset: %.2f", mOffset);
        pt = new Point(100, 500);
        Imgproc.putText(outImg, label, pt, fontFace, scale, new Scalar(220, 220, 255), thickness);

        label = String.format("radius: %.2f", mCurvatureRadius);
        pt = new Point(100, 550);
        Imgproc.putText(outImg, label, pt, fontFace, scale, new Scalar(220, 220, 255), thickness);

        // memory release
        warpImg.release();
    }


    /**
     * Project image into bird's eye view with Homograph transform
     *
     * @param img input image
     * @param dst warped image
     * @param M   output transform matrix
     *            Homograph Transform equation from a to b:
     *            p_a = (z_b/z_a)*K_a*H*K_b^-1*p_b;
     *            H = R - T; R = R_a*R_b^t, T = t*n^t/d; t = R_a*R_b^t*t_b-t_a; // n in b coordinates
     *            distance = cos(vertical_ang)*(1/cos(vertical_ang-pitch) // keep original distance
     *            delta = distance*tan^-1(vertical_ang2) - tan^-1(vertical_ang - pitch) // align to bottom of the frame
     *            In our case: R = [-pitch][-roll], t = R*(0, -delta, 1-distance); n = (0, 0, -1);
     *            => H = R[1, 0, 0; 0, 1, -delta/distance; 0, 0, 1/distance]
     */
    public void birdsEyeTransform(Mat img, Mat dst, Mat M) {
        double distance = Math.cos(mSensorVerticalAngle / 2.f) // distance of source camera from plain
                * (1 / Math.cos(mSensorVerticalAngle / 2.f - mPitchAngle));
        double delta = (distance * Math.tan(mLanesSensorVerticalAngle / 2.f)) // bird's eye sensor edge x-distance from camera
                - (Math.tan(mSensorVerticalAngle / 2.f - mPitchAngle)); // original sensor edge x-distance from camera
        Mat R = ImageOperations.setRotationMatrix(-mPitchAngle, "x");
        Mat H = Mat.eye(3, 3, CvType.CV_64FC1);
        H.put(1, 2, -delta / distance);
        H.put(2, 2, 1 / distance);
        Core.gemm(R, H, 1, new Mat(), 0, H);
        Core.gemm(mCameraMatrix, H, 1, new Mat(), 0, M);
        Core.gemm(M, mLanesCameraMatrix.inv(), 1, new Mat(), 0, M);

        Core.invert(M, M);

        Imgproc.warpPerspective(img, dst, M, mLanesProjectedSize);

        R.release();
        H.release();
    }


    /**
     * Find region of interest for lanes by rotation matrix and vanishing point of closer lines
     *
     * @param img            input image in BW format (uint8)
     * @param vanishingPoint output vanishing point
     * @param drawLines      draw found lines
     */
    private void findVanishingPoint(Mat img, Point vanishingPoint, boolean drawLines) {

        // build windows to detect the two lines in front of the car
        int margin = 3 * img.cols() / 32;
        int winyLow = 12 * img.rows() / 16;
        int winyHigh = img.rows();
        int winxLeftLow = img.width() / 4 - margin;
        int winxLeftHigh = img.width() / 4 + margin;
        int winxRightLow = 3 * img.width() / 4 - margin;
        int winxRightHigh = 3 * img.width() / 4 + margin;

        Mat leftFrontCar = img.submat(winyLow, winyHigh, winxLeftLow, winxLeftHigh);
        Mat rightFrontCar = img.submat(winyLow, winyHigh, winxRightLow, winxRightHigh);

//		Mat leftFrontCarFiltered = new Mat();
//		Mat rightFrontCarFiltered = new Mat();
//		filterLines(leftFrontCar, leftFrontCarFiltered, mLeftAngle);
//		filterLines(rightFrontCar, rightFrontCarFiltered, mRightAngle);

        // calculate polynomial constants with last frames mean
        Mat betaL = new Mat(2, 1, CvType.CV_64FC1), betaR = betaL.clone();
        // FitLine
        boolean left = findLinesFitLine(leftFrontCar, betaL, winxLeftLow, winyLow, true, drawLines);
        boolean right = findLinesFitLine(rightFrontCar, betaR, winxRightLow, winyLow, false, drawLines);
        // Hough Line
//        boolean left = findLinesHough(leftFrontCar, betaL, winxLeftLow, winyLow, true, drawLines);
//        boolean right = findLinesHough(rightFrontCar, betaR, winxRightLow, winyLow, false, drawLines);

        if (left && right) {
            if (mLinearCounter < NUM_FRAMES) {
                List<Mat> src = Arrays.asList(mLinearBetasL, betaL);
                Core.hconcat(src, mLinearBetasL);
                src = Arrays.asList(mLinearBetasR, betaR);
                Core.hconcat(src, mLinearBetasR);
            } else {
                betaL.copyTo(mLinearBetasL.col(mLinearCounter % NUM_FRAMES));
                betaR.copyTo(mLinearBetasR.col(mLinearCounter % NUM_FRAMES));
            }
            mLinearCounter++;
        }

        if (mLinearBetasL.cols() > 0 && mLinearBetasR.cols() > 0) {
//            // Average
//			Core.reduce(mLinearBetasL, betaL, 1, Core.REDUCE_AVG);
//			Core.reduce(mLinearBetasR, betaR, 1, Core.REDUCE_AVG);
            // Median
            Mat sortedL = new Mat(), sortedR = new Mat();
            Core.sort(mLinearBetasL, sortedL, Core.SORT_EVERY_ROW);
            Core.sort(mLinearBetasR, sortedR, Core.SORT_EVERY_ROW);
            betaL = sortedL.col(sortedL.cols() / 2);
            betaR = sortedR.col(sortedR.cols() / 2);
        }

        // find vanishing point
        Mat A = new Mat(2, 2, CvType.CV_64FC1), yx = new Mat(), b = new Mat(2, 1, CvType.CV_64FC1);
        A.put(0, 0, betaL.get(1, 0)[0]);
        A.put(1, 0, betaR.get(1, 0)[0]);
        A.put(0, 1, -1);
        A.put(1, 1, -1);
        b.put(0, 0, -betaL.get(0, 0)[0]);
        b.put(1, 0, -betaR.get(0, 0)[0]);
        Core.solve(A, b, yx);
        vanishingPoint.x = yx.get(1, 0)[0];
        vanishingPoint.y = yx.get(0, 0)[0];

        // draw the vanishing point
        Imgproc.circle(img, vanishingPoint, 5, new Scalar(128), -1);

    }


    /**
     * Filter image to get thin lines
     *
     * @param img   input image in BW format (uint8)
     * @param dst   output filtered image
     * @param angle expected lines direction in radians
     */
    private void filterLines(Mat img, Mat dst, double angle) {

        Mat morph = Mat.zeros(11, 11, CvType.CV_8UC1);
        morph.col(5).setTo(new Scalar(255));
        Mat morphRot = new Mat();

        Imgproc.warpAffine(morph, morphRot, Imgproc.getRotationMatrix2D(new Point(5, 5), angle, 1.),
                new Size(11, 11));
        Imgproc.morphologyEx(img, dst, Imgproc.MORPH_ERODE, morph);

        Mat kernel = new Mat(3, 3, CvType.CV_8SC1);
        byte[] data = {-1, 0, 1, 0, 0, 0, -1, 0, 1};
        kernel.put(0, 0, data);

        if (angle > 0)
            Core.flip(kernel, kernel, 1);

        Imgproc.filter2D(dst, dst, -1, kernel);

    }


    /**
     * Find straight lines with HoughLine algorithm
     *
     * @param img     input image in BW format (uint8)
     * @param beta    output vector of line parameters (b0 + b1*x)
     * @param offsetX offset to add in x axis (if the image is part of bigger image)
     * @param offsetY offset to add in y axis (if the image is part of bigger image)
     * @param dir     direction of expected line: left to right - true, right to left - false
     * @param dir     add the found lines to the image
     */
    private boolean findLinesHough(Mat img, Mat beta, int offsetX, int offsetY, boolean dir,
                                   boolean drawLines) {

        List<Mat> coordinates = new ArrayList<>();
        Mat xSub = new Mat(), ySub = new Mat(), magnitudes = new Mat();
        Mat linesP = new Mat(); // will hold the results of the detection

        // Detect left and right lines
        Imgproc.HoughLinesP(img, linesP, 1, Math.PI / 180, 1, 10, 8);

        if (linesP.rows() > 0) {

            // find the longest line
            Core.split(linesP, coordinates);
            Core.subtract(coordinates.get(1), coordinates.get(0), xSub, new Mat(), CvType.CV_64FC1);
            Core.subtract(coordinates.get(3), coordinates.get(2), ySub, new Mat(), CvType.CV_64FC1);
            Core.magnitude(xSub, ySub, magnitudes);
            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(magnitudes);
            double[] l = linesP.get((int) minMaxLocResult.maxLoc.y, 0);

            // Keep lines coordinates
            Mat laneInds = new Mat(2, 1, CvType.CV_32SC2);
            laneInds.put(0, 0, l);
            Core.add(laneInds, new Scalar(offsetX, offsetY), laneInds);

            // Fit a linear polynomial
            fitPoly(laneInds, 1, true, beta, false);

            double angle = calculatePolyAngle(0, beta);

            // check reasonable direction
            if ((angle < -Math.PI / 8 && angle > -3 * Math.PI / 8 && dir)
                    || (angle < 3 * Math.PI / 8 && angle > Math.PI / 8 && !dir)) {
                if (dir) {
                    mLeftAngle = angle;
                } else {
                    mRightAngle = angle;
                }
                if (drawLines) {
                    Imgproc.line(img, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(100), 3, Imgproc.LINE_AA,
                            0);
                }
                return true;
            }
        }

        return false;
    }


    /**
     * Find straight lines with FitLine algorithm
     *
     * @param img     input image in BW format (uint8)
     * @param beta    output vector of line parameters (b0 + b1*x)
     * @param offsetX offset to add in x axis (if the image is part of bigger image)
     * @param offsetY offset to add in y axis (if the image is part of bigger image)
     * @param dir     direction of expected line: left to right - true, right to left - false
     * @param dir     add the found lines to the image
     */
    private boolean findLinesFitLine(Mat img, Mat beta, int offsetX, int offsetY, boolean dir,
                                     boolean drawLines) {

        MatOfPoint goodInds = new MatOfPoint();
        Core.findNonZero(img, goodInds);

        if (goodInds.rows() > 10) {

            // fit line using FitLine algorithm
            Mat line = new Mat();
            Imgproc.fitLine(goodInds, line, Imgproc.CV_DIST_HUBER, 0, 0.01, 0.01);

            // extract polynomial parameters: x = m*y + b
            float[] l = new float[4];
            line.get(0, 0, l);
            double m = l[0] / l[1];
            double b = l[2] - m * l[3];
            double angle = Math.atan(m);

            // set output polynomial parameters
            beta.put(0, 0, b + offsetX - m * (offsetY));
            beta.put(1, 0, m);

            // check reasonable direction
            if ((angle < -Math.PI / 8 && angle > -3 * Math.PI / 8 && dir)
                    || (angle < 3 * Math.PI / 8 && angle > Math.PI / 8 && !dir)) {
                if (dir) {
                    mLeftAngle = angle;
                } else {
                    mRightAngle = angle;
                }
                if (drawLines) {
                    // double x_top = b;
                    double x_bottom = m * (double) (img.height() - 1) + b;
                    Imgproc.line(img, new Point(b, 0), new Point(x_bottom, img.height() - 1), new Scalar(100), 3,
                            Imgproc.LINE_AA, 0);
                }
                return true;
            }
        }

        return false;
    }


    /**
     * Filter input image for pre-detecting lanes
     *
     * @param src input image in RGB format.
     * @param dst filtered output image
     */
    private static void filterPipeline(Mat src, Mat dst) {

        Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(dst, dst, new Size(7, 7), 0, 0);
        Imgproc.threshold(dst, dst, 0, 255,
                Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
        if (Core.countNonZero(dst) > dst.height() * dst.width() / 2) // if background is white
            Core.bitwise_not(dst, dst); // revert

//        // erode filtered lines
//        Imgproc.Sobel(dst, dst, -1, 1, 0);

//        Imgproc.Sobel(dst.colRange(0, dst.width()/2), dst.colRange(0, dst.width()/2),
//                -1, -1, 0);
//        Imgproc.Sobel(dst.colRange(dst.width()/2, dst.width()), dst.colRange(dst.width()/2, dst.width()),
//                -1, 1, 0);
//        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7,7));
//        Imgproc.morphologyEx(mMask, mMask, Imgproc.MORPH_OPEN, kernel);
//        kernel.release();
    }


    /**
     * find lanes on black and white filtered bird's eye image with histogram and sliding windows
     * method, return steering value
     *
     * @param img    input image in black and white format.
     * @param outImg grayscale picture of detected road and sliding windows,
     *               should be same size of img and initiated with zeros.
     */
    private void lanesDetection(Mat img, Mat outImg) {

        // Create empty lists to receive left and right lane pixel indices
        MatOfPoint leftLaneInds = new MatOfPoint();
        MatOfPoint rightLaneInds = new MatOfPoint();

        // Make a histogram of each column in the image
        Mat histogram = new Mat();
        Core.reduce(img, histogram, 0, Core.REDUCE_SUM, CvType.CV_32FC1);

        // find start of lanes - Otsu's threshold
        int[] xBases = new int[2];
        int midpoint = findOtsuThreshold(histogram, histogram, xBases);
        int xBaseLeft = xBases[0], xBaseRight = xBases[1];

//        int midpoint = img.width() / 2;
//
//        // find start of lanes - peaks
//        Core.MinMaxLocResult minMaxLocResult;
//        minMaxLocResult = Core.minMaxLoc(img.colRange(0, midpoint));
//        int xBaseLeft = (int) minMaxLocResult.maxLoc.x;
//        minMaxLocResult = Core.minMaxLoc(img.colRange(midpoint, img.width()));
//        int xBaseRight = (int) minMaxLocResult.maxLoc.x;
//
//        // find start of lanes - big windows
//        Mat boundingMatLeft = img.submat(img.height() / 8, img.height(), 0, midpoint);
//        Mat boundingMatRight = img.submat(img.height() / 8, img.height(), midpoint, img.width());
//        Mat nonZeroPointsLeft = new Mat();
//        Mat nonZeroPointsRight = new Mat();
//        Core.findNonZero(boundingMatLeft, nonZeroPointsLeft);
//        Core.findNonZero(boundingMatRight, nonZeroPointsRight);
//        int xBaseLeft = (int) Core.mean(nonZeroPointsLeft).val[0];
//        int xBaseRight = (int) Core.mean(nonZeroPointsRight).val[0];

        Size winSize = new Size(img.width() / 4, img.height() / 16);
        // find lanes with sliding windows algorithm
        mLeftLaneDir = slidingWindows(img, outImg, winSize, 24, 10,
                xBaseLeft, true, leftLaneInds);
        mRightLaneDir = slidingWindows(img, outImg, winSize, 24, 10,
                xBaseRight, true, rightLaneInds);

        // Fit a second order polynomial to each
        Mat betaL = new Mat(), betaR = new Mat();
        if (leftLaneInds.rows() > POLY_ORDER + 1 && rightLaneInds.rows() > POLY_ORDER + 1) {

            fitPoly(leftLaneInds, POLY_ORDER, true, betaL, true);
            fitPoly(rightLaneInds, POLY_ORDER, true, betaR, true);

            // calculate polynomial constants with last frames mean
            if (mPolyCounter < NUM_FRAMES) {
                List<Mat> src = Arrays.asList(mPolyBetasL, betaL);
                Core.hconcat(src, mPolyBetasL);
                src = Arrays.asList(mPolyBetasR, betaR);
                Core.hconcat(src, mPolyBetasR);
            } else {
                betaL.copyTo(mPolyBetasL.col(mPolyCounter % NUM_FRAMES));
                betaR.copyTo(mPolyBetasR.col(mPolyCounter % NUM_FRAMES));
            }
            mPolyCounter++;

        }

        // draw the road in the out image
        Mat road = new Mat(img.size(), CvType.CV_8U, Scalar.all(0));
//        fillPolyCurves(betaL, betaR, road, 8);
        fillCurves(leftLaneInds, rightLaneInds, road);

        // calculate offset - with detected points
        if (!leftLaneInds.empty() && !rightLaneInds.empty()) {
            double carPos = img.width() / 2;
            double laneCenterPos = (leftLaneInds.get(0, 0)[0] +
                    rightLaneInds.get(0, 0)[0]) / 2;
            mOffset = carPos - laneCenterPos;
        }


        // draw all in out image
        Core.bitwise_or(outImg, road, outImg);

        // memory release
        histogram.release();
        betaL.release();
        betaR.release();
        leftLaneInds.release();
        rightLaneInds.release();
        road.release();
    }


    /**
     * find lanes on black and white filtered bird's eye image with histogram and sliding windows
     * method, return steering value
     *
     * @param img       input grayscale image of lanes
     * @param outImg    output grayscale image with direction arrow
     * @param signNames list of founded signs in the frame
     */
    public void calculateDriving(Mat img, Mat outImg, ArrayList<String> signNames) {

        double lanesAngle = 0;
        double steeringAngle = 0;
        Mat betaL = new Mat();
        Mat betaR = new Mat();

        decipherSigns(signNames);

        // calculate steering
        if (mPolyBetasL.cols() > 0 && mPolyBetasR.cols() > 0) { // There is valid betaL and betaR
            Core.reduce(mPolyBetasL, betaL, 1, Core.REDUCE_AVG);
            Core.reduce(mPolyBetasR, betaR, 1, Core.REDUCE_AVG);

//            // calculate offset - with polynomial
            double carPos = img.width() / 2;
//            double laneCenterPos = (calculatePoly(img.height() - 1, betaL) +
//                    calculatePoly(img.height() - 1, betaR)) / 2;
//            mOffset = carPos - laneCenterPos;

            boolean junction = Math.abs(mLeftLaneDir[0] - mRightLaneDir[0]) > 0.5;
            // junction hysteresis
            if (junction && mJunctionCounter < 10)
                mJunctionCounter++;
            else if (!junction && mJunctionCounter > 0)
                mJunctionCounter--;
            if (!mIsJunction && mJunctionCounter > 7) // no junction -> junction
                mIsJunction = true;
            else if (mIsJunction && mJunctionCounter < 3) { // junction -> no junction
                mIsJunction = false;
                mNextTurn = Direction.STRAIGHT;
            }
            if (mIsJunction) {
                Imgproc.putText(outImg, "junction",
                        new Point(outImg.width() / 2 - 72, outImg.height() / 2), 1,
                        4, new Scalar(255));
                switch (mNextTurn) {
                    case LEFT: {
                        lanesAngle = -calculatePolyAngle(img.height() - 1, betaL);
                        mCurvatureRadius = calculatePolyCurveRadius(img.height() - 1, betaL);
                        break;
                    }
                    case RIGHT: {
                        lanesAngle = -calculatePolyAngle(img.height() - 1, betaR);
                        mCurvatureRadius = calculatePolyCurveRadius(img.height() - 1, betaR);
                        break;
                    }
                    case STRAIGHT: {
                        lanesAngle = -(calculatePolyAngle(img.height() - 1, betaL) +
                                calculatePolyAngle(img.height() - 1, betaR)) / 2;
                        mCurvatureRadius = (calculatePolyCurveRadius(img.height() - 1, betaL) +
                                calculatePolyCurveRadius(img.height() - 1, betaR)) / 2;
                    }
                }
            } else {
                lanesAngle = -(calculatePolyAngle(img.height() - 1, betaL) +
                        calculatePolyAngle(img.height() - 1, betaR)) / 2;
                mCurvatureRadius = (calculatePolyCurveRadius(img.height() - 1, betaL) +
                        calculatePolyCurveRadius(img.height() - 1, betaR)) / 2;
            }

            int arrowMaxLen = img.height() / 2;
            steeringAngle = -Math.atan(mOffset / arrowMaxLen);
//            steeringAngle = lanesAngle - Math.atan(mOffset / img.height());
            double power = 150 * Math.cos(steeringAngle);
            mPower = power < mSpeedLimit ? (int) Math.round(power / 1.5)
                    : (int) Math.round(mSpeedLimit / 1.5);

            // draw direction arrow
            img.copyTo(outImg);
            int arrowEndY = (int) (img.height() - (mPower / 100.0 * arrowMaxLen) * Math.cos(steeringAngle));
            int arrowEndX = (int) (carPos + ((mPower / 100.0 * arrowMaxLen) * Math.sin(steeringAngle)));
            Imgproc.arrowedLine(outImg, new Point(carPos, img.height() - 1),
                    new Point(arrowEndX, arrowEndY), new Scalar(0), 12);

        }

        mSteering = (int) (100 * (steeringAngle / (2 * Math.PI)));

        // memory release
        betaL.release();
        betaR.release();

    }


    /**
     * find Otsu threshold for given input histogram (minimization of intra-class variance)
     *
     * @param histogram     input one row histogram mat of float
     * @param normHistogram output one row normalized histogram mat of bytes
     * @param means         output int array of size 2 for 2 means values of the 2 classes
     */
    public int findOtsuThreshold(Mat histogram, Mat normHistogram, int[] means) {

        // normalize histogram
        Scalar scale = new Scalar(1 / Core.sumElems(histogram).val[0]);
        Core.multiply(histogram, scale, normHistogram);

        // pour mat into array for performance
        int histSize = normHistogram.width();
        float[] histArr = new float[histSize];
        normHistogram.get(0, 0, histArr);

        // initiate histogram LUTs
        float[] cumHist = new float[histSize]; // cumulative histogram probabilities
        cumHist[0] = histArr[0];
        float[] cumWeightProbabilities = new float[histSize]; // cumulative histogram probabilities
        cumWeightProbabilities[0] = 0;
        for (int i = 1; i < histSize; i++) {
            cumHist[i] = cumHist[i - 1] + histArr[i];
            cumWeightProbabilities[i] = cumWeightProbabilities[i - 1] + (i * histArr[i]);
        }

        // Otsu's method variables
        float mu0, mu1, w0, w1, sigmab, max = 0;
        int maxIdx = histSize / 2;

        for (int i = 0; i < histSize; i++) {
            // find weights
            w0 = cumHist[i];
            w1 = 1 - w0;
            if (w0 == 0 || w1 == 0) // sigmab == 0
                continue;
            // find means
            mu0 = cumWeightProbabilities[i] / w0;
            mu1 = (cumWeightProbabilities[histSize - 1] - cumWeightProbabilities[i]) / w1;
            // find inter-class variance
            sigmab = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
            // max inter-class equal to min intra-class
            if (sigmab > max) {
                max = sigmab;
                maxIdx = i;
            }
        }

        means[0] = Math.round(cumWeightProbabilities[maxIdx] / cumHist[maxIdx]);
        means[1] = Math.round((cumWeightProbabilities[histSize - 1] - cumWeightProbabilities[maxIdx]) / (1 - cumHist[maxIdx]));
        return maxIdx;
    }


    /**
     * find lanes on black and white filtered bird's eye image with histogram and sliding windows
     * method. returns last window direction vector (vx, vy)
     *
     * @param img         input image in black and white format.
     * @param outImg      image
     * @param winSize     number of windows for sliding windows algorithm
     * @param maxWindows  maximum number of windows for sliding
     * @param minPix      minimum number of pixels per window
     * @param drawWindows draw the sliding windows
     * @param laneInds    detected points indices of the lane
     */
    private double[] slidingWindows(Mat img, Mat outImg, Size winSize, int maxWindows, int minPix,
                                    int xBase, boolean drawWindows, MatOfPoint laneInds) {
        // window variables
        RotatedRect window = new RotatedRect(new Point(xBase, img.height() - winSize.height / 2),
                winSize, 0);
        Point[] windowPoints = new Point[4];
        Mat windowMask = new Mat();
        Scalar white = new Scalar(255);
        List<MatOfPoint> croppedWindows = new ArrayList<>();
        int windowCounter = 0;
        double iou = 1.0;

        // lane variables
        List<Point> lanePoints = new ArrayList<>();
        MatOfPoint nonZeroPoints = new MatOfPoint();
        MatOfPoint mop = new MatOfPoint(new Point()); // helper Mat Of Point
        double[] means = {xBase, (img.height() + winSize.height / 2)};
        double[] new_means;

        // rotation variables
        Mat line = new Mat(4, 1, CvType.CV_32FC1);
        // line parameters [vx, vy, x, y]
        float[] l = {0, 1, xBase, (float) (img.height() - winSize.height / 2)};
        line.put(0, 0, l);
        double vy = -1, vx = 0;

        // Step through the windows one by one
        do {
            // crop window with image parameters
            window.points(windowPoints);
            for (int i = 0; i < 4; i++) {
                windowPoints[i].x = windowPoints[i].x < 0 ? 0 :
                        (windowPoints[i].x >= img.width() ? img.width() - 1 : windowPoints[i].x);
                windowPoints[i].y = windowPoints[i].y < 0 ? 0 :
                        (windowPoints[i].y >= img.height() ? img.height() - 1 : windowPoints[i].y);
            }
            MatOfPoint windowCropped = new MatOfPoint(windowPoints);
            croppedWindows.add(windowCropped);
            // calculate intersect over union of contour with image
            iou = Imgproc.contourArea(windowCropped) / (winSize.width * winSize.height);

            windowCounter++;

            Rect boundingRect = Imgproc.boundingRect(windowCropped);
            Scalar topRect = new Scalar(boundingRect.x, boundingRect.y);

            // create mask from the rotated window
            windowMask.create(boundingRect.size(), CvType.CV_8U);
            Core.subtract(windowCropped, topRect, mop);
            Imgproc.fillConvexPoly(windowMask, mop, white);
            // filter only the points within the mask
            Mat boundingMat = img.submat(boundingRect).clone();
            Core.bitwise_and(boundingMat, windowMask, boundingMat);

            // Identify the nonzero pixels in x and y within the window
            Core.findNonZero(boundingMat, nonZeroPoints);
            Core.add(nonZeroPoints, topRect, nonZeroPoints);

            // If founded > minPix pixels
            if (nonZeroPoints.height() > minPix) {
                // recenter next window on their mean position
                new_means = Core.mean(nonZeroPoints).val;
                lanePoints.add(new Point(new_means));
                vy = (new_means[1] - means[1]);
                vx = (new_means[0] - means[0]);
                double norm = Math.sqrt(vx * vx + vy * vy);
                vy = vy / norm;
                vx = vx / norm;
                means[0] = new_means[0];
                means[1] = new_means[1];
            } else {
                means[0] = window.center.x;
                means[1] = window.center.y;
            }
            // prepare for next iteration
            window.angle = Math.toDegrees(-Math.atan(vx / vy));
            window.center.x = vx * winSize.height + means[0];
            window.center.y = vy * winSize.height + means[1];

            boundingMat.release();

        }
        while (iou > 0.25 && maxWindows > windowCounter);


        if (drawWindows == true)
            Imgproc.drawContours(outImg, croppedWindows, -1, white);

        laneInds.fromList(lanePoints);

        // memory release
        windowMask.release();
        for (MatOfPoint win : croppedWindows)
            win.release();
        nonZeroPoints.release();
        mop.release();
        line.release();

        // calculate lane direction
        double vX = 0, vY = 1;
        if (!lanePoints.isEmpty()) {
            vX = lanePoints.get(lanePoints.size() - 1).x - lanePoints.get(0).x;
            vY = lanePoints.get(lanePoints.size() - 1).y - lanePoints.get(0).y;
            double norm = Math.sqrt(vX * vX + vY * vY);
            vY = vY / norm;
            vX = vX / norm;
        }
        double[] dirVec = {vX, vY};

        return dirVec;

    }

    /**
     * Set the class sensor info for image warping
     *
     * @param focalLength       focal length in mm
     * @param sensorWidth       width of camera sensor in mm
     * @param sensorHeight      height of camera sensor in mm
     * @param frameWidth        width of input image in pixels
     * @param frameHeight       height of input image in pixels
     * @param lanesFrameWidth   width of transformed lanes image
     * @param lanesFrameHeight  height of transformed lanes image
     * @param sensorWidthScale  scale of lanes camera sensor width relative to input image camera
     */
    public void setSensorInfo(float focalLength, float sensorWidth, float sensorHeight, int frameWidth,
                              int frameHeight, int lanesFrameWidth, int lanesFrameHeight, float sensorWidthScale) {
        // original sensor
        mFocalLength = focalLength;
        mSensorWidth = sensorWidth;
        mSensorHeight = sensorHeight;
        mSensorHorizontalAngle = (float) (2 * Math.atan(mSensorWidth / (mFocalLength * 2)));
        mSensorVerticalAngle = (float) (2 * Math.atan(mSensorHeight / (mFocalLength * 2)));

        float fx = mFocalLength * frameWidth / mSensorWidth;
        float fy = mFocalLength * frameHeight / mSensorHeight;
        double intrinsic[] = {fx, 0, frameWidth / 2, 0, fy, frameHeight / 2, 0, 0, 1};
        mCameraMatrix.put(0, 0, intrinsic);

        // transformed lanes camera sensor
        mLanesProjectedSize = new Size(lanesFrameWidth, lanesFrameHeight);
        mLanesFocalLength = mFocalLength;
        mLanesSensorWidth = mSensorWidth * sensorWidthScale;
        float heightWidthScale = (mSensorHeight / (float) frameHeight) / (mSensorWidth / (float) frameWidth);
        mLanesSensorHeight = heightWidthScale * mSensorWidth * (lanesFrameHeight / (float) lanesFrameWidth);
        mLanesSensorHorizontalAngle = (float) (2 * Math.atan(mLanesSensorWidth / (mLanesFocalLength * 2)));
        mLanesSensorVerticalAngle = (float) (2 * Math.atan(mLanesSensorHeight / (mLanesFocalLength * 2)));

        float fx2 = mLanesFocalLength * lanesFrameWidth / mLanesSensorWidth;
        float fy2 = mLanesFocalLength * lanesFrameHeight / mLanesSensorHeight;
        double intrinsic2[] = {fx2, 0, lanesFrameWidth / 2, 0, fy2, lanesFrameHeight / 2, 0, 0, 1};
        mLanesCameraMatrix.put(0, 0, intrinsic2);
    }

    /**
     * fit a polynomial for a given set of points by regression.
     *
     * @param curve      set of point of the curve to fit
     * @param degree     the degree of the fitted polynomial
     * @param y_function is x is a function of y or the opposite
     * @param beta       output vector of the coefficients of the polynomial
     */
    public static void fitPoly(Mat curve, int degree, Boolean y_function, Mat beta,
                               boolean weighted) {

        int n = curve.rows();
        int l = degree + 1;

        Mat matrixX = new Mat(n, l, CvType.CV_64FC1);
        Mat vectorY = new Mat(n, 1, CvType.CV_64FC1);
        Mat vectorX = new Mat(n, 1, CvType.CV_64FC1);

        List<Mat> xys = new ArrayList<>();
        Core.split(curve, xys);

        // build matrices to solve the regression problem
        if (y_function) { // x = f(y)
            xys.get(0).convertTo(vectorY, CvType.CV_64FC1);
            xys.get(1).convertTo(vectorX, CvType.CV_64FC1);
        } else {          // y = f(x)
            xys.get(1).convertTo(vectorY, CvType.CV_64FC1);
            xys.get(0).convertTo(vectorX, CvType.CV_64FC1);
        }
        double[] buffVecX = new double[n];
        double[] buffVecY = new double[n];
        vectorX.get(0, 0, buffVecX);
        if (weighted)
            vectorY.get(0, 0, buffVecY);

        // fill the X matrix according the given functions
        double[] buffMatX = new double[n * l];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= degree; j++)
                buffMatX[i * l + j] = weighted ? (n - i) * Math.pow(buffVecX[i], j) :
                        Math.pow(buffVecX[i], j);
            if (weighted)
                buffVecY[i] = buffVecY[i] * (n - i);
        }
        matrixX.put(0, 0, buffMatX);
        if (weighted)
            vectorY.put(0, 0, buffVecY);

        Core.solve(matrixX, vectorY, beta, Core.DECOMP_QR);

        // memory release
        vectorX.release();
        vectorY.release();
        matrixX.release();
        xys.get(0).release();
        xys.get(1).release();

    }
//
//
//    /**
//     * fit a polynomial for a given set of points by regression. using apache library.
//     *
//     * @param curve      set of point of the curve to fit
//     * @param degree     the degree of the fitted polynomial
//     * @param y_function is x is a function of y or the opposite
//     * @param beta       output vector of the coefficients of the polynomial
//     */
//    public static void fitPolyApache(Mat curve, int degree, Boolean y_function, Mat beta) {
//
//        int n = curve.rows();
//        int[] vectorX = new int[n];
//        int[] vectorY = new int[n];
//
//        List<Mat> xys = new ArrayList<>();
//        Core.split(curve, xys);
//
//        // build matrices to solve the regression problem
//        if (y_function) { // x = f(y)
//            xys.get(0).get(0, 0, vectorY);
//            xys.get(1).get(0, 0, vectorX);
//        } else {          // y = f(x)
//            xys.get(1).get(0, 0, vectorY);
//            xys.get(0).get(0, 0, vectorX);
//        }
//
//        final WeightedObservedPoints obs = new WeightedObservedPoints();
//        for (int i = 0; i < n; i++) {
//            obs.add(vectorX[i], vectorY[i]);
//        }
//
//        double[] coeff = new double[degree + 1]; // initiate with zeros
//        coeff[0] = vectorY[0];
//        PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree).withStartPoint(coeff);
//        coeff = fitter.fit(obs.toList());
//
//
//        beta.create(degree + 1, 1, CvType.CV_64FC1);
//        beta.put(0, 0, coeff);
//
//        // memory release
//        xys.get(0).release();
//        xys.get(1).release();
//
//
//    }

    /**
     * Calculate and return the polynomial function on a given point
     *
     * @param x    input point
     * @param beta input vector of the coefficients of the polynomial
     */
    public static double calculatePoly(double x, Mat beta) {
        int n = beta.rows() - 1;
        double y = 0;
        for (int i = 0; i < n; i++) {
            y += beta.get(i, 0)[0] * Math.pow(x, i);
        }
        return y;
    }


    /**
     * Fill in the given image the area between two polynomials curves
     *
     * @param betaL input vector of the coefficients of the polynomial left curve
     * @param betaR input vector of the coefficients of the polynomial right curve
     * @param img   image to draw inside
     * @param step  the distance in pixels between two points of the curves to fill (speed <-> accuracy)
     */
    private static void fillPolyCurves(Mat betaL, Mat betaR, Mat img, int step) {

        Point[] drawPoints1 = new Point[img.rows() / step + 1];
        Point[] drawPoints2 = new Point[img.rows() / step + 1];
        int n = drawPoints1.length - 1;

        for (int i = 0, ind = 0; ind < n; i += step, ind++) {
            drawPoints1[ind] = new Point(calculatePoly(i, betaL), i);
            drawPoints2[ind] = new Point(calculatePoly(i, betaR), i);
        }
        // fill bottom of the image
        drawPoints1[n] = new Point(calculatePoly(img.rows() - 1, betaL),
                img.rows() - 1);
        drawPoints2[n] = new Point(calculatePoly(img.rows() - 1, betaR),
                img.rows() - 1);

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

        Imgproc.fillPoly(img, ppt, new Scalar(128));

        // memory release
        curve1.release();
        curve2.release();

    }


    /**
     * Fill in the given image the area between two polynomials curves
     *
     * @param curveL input points of left curve
     * @param curveR input points of right curve
     * @param img    image to draw inside
     */
    private static void fillCurves(MatOfPoint curveL, MatOfPoint curveR, Mat img) {

        if (curveL.checkVector(2, CvType.CV_32S) < 0 ||
                curveR.checkVector(2, CvType.CV_32S) < 0)
            return;

        // flip right curve
        MatOfPoint fullPolygon = new MatOfPoint();
        Core.flip(curveR, fullPolygon, 0);

        // create shape and fill it.
        fullPolygon.push_back(curveL);
        List<MatOfPoint> ppt = new ArrayList<>();
        ppt.add(fullPolygon);

        Imgproc.fillPoly(img, ppt, new Scalar(128));

        // memory release
        fullPolygon.release();

    }


    /**
     * calculate the slope angle on a given point
     *
     * @param x    input point
     * @param beta input vector of the coefficients of the polynomial function
     */
    public static double calculatePolyAngle(double x, Mat beta) {
        Mat betaDiv = new Mat();
        polyDiv(beta, betaDiv);
        double dfdx = calculatePoly(x, betaDiv);
        return Math.atan(dfdx);
    }


    /**
     * create the derivative on a given point
     *
     * @param beta    input vector of the coefficients of the polynomial function
     * @param betaDiv output vector of the coefficients of the derivative polynomial function
     */
    public static void polyDiv(Mat beta, Mat betaDiv) {
        int n = beta.rows();
        betaDiv.create(n - 1, 1, CvType.CV_64FC1);
        double[] divCoeff = new double[n];
        for (int i = 1; i < n; i++) {
            divCoeff[i - 1] = beta.get(i, 0)[0] * i;
        }
        betaDiv.put(0, 0, divCoeff);
    }

    /**
     * calculate the curvature radius on a given point (R = (1+d(f/dx)^2)^1.5/|d2fdx2|)
     *
     * @param x    input point
     * @param beta input vector of the coefficients of the polynomial function
     */
    public static double calculatePolyCurveRadius(double x, Mat beta) {
        Mat betaDiv = new Mat();
        Mat betaDiv2 = new Mat();

        polyDiv(beta, betaDiv);
        polyDiv(betaDiv, betaDiv2);

        double dfdx = calculatePoly(x, betaDiv);
        double dfdx2 = calculatePoly(x, betaDiv2);

        double r = Math.pow(1 + 2 * (dfdx * dfdx), 1.5);
        r = r / Math.abs(dfdx2);

        return r;

    }


    /**
     * Decipher traffic sign string into power, steering and next turn.
     *
     * @param signNames input array of strings of traffic sign names
     */
    public void decipherSigns(ArrayList<String> signNames) {
        for (String signName : signNames) {
            switch (signName) {
                case "speed limit 20": {
                    mSpeedLimit = 20;
                    break;
                }
                case "speed limit 30": {
                    mSpeedLimit = 30;
                    break;
                }
                case "speed limit 50": {
                    mSpeedLimit = 50;
                    break;
                }
                case "speed limit 60": {
                    mSpeedLimit = 60;
                    break;
                }
                case "speed limit 70": {
                    mSpeedLimit = 70;
                    break;
                }
                case "speed limit 80": {
                    mSpeedLimit = 80;
                    break;
                }
                case "speed limit 100": {
                    mSpeedLimit = 100;
                    break;
                }
                case "speed limit 120": {
                    mSpeedLimit = 120;
                    break;
                }
                case "restriction ends 80": {
                    mSpeedLimit = 150;
                    break;
                }
                case "go left": {
                    mNextTurn = Direction.LEFT;
                    break;
                }
                case "go right": {
                    mNextTurn = Direction.RIGHT;
                    break;
                }
                case "stop": {
                    mPower = 0;
                    break;
                }
            }
        }
    }


    public enum Direction {
        LEFT, RIGHT, STRAIGHT
    }

}

