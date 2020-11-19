package com.autonomousphonecar;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.util.Log;
import android.util.SizeF;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";

    {
//        System.loadLibrary("opencv_java");
    }

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }


    public static final int VIEW_MODE_RGBA = 0;
    public static final int VIEW_MODE_COLOR = 1;
    public static final int VIEW_MODE_SHAPE = 2;
    public static final int VIEW_MODE_SIGN = 3;
    public static final int VIEW_MODE_RECT = 4;
    public static final int VIEW_MODE_LANE = 5;
    public static final int VIEW_MODE_NET = 6;
    public static final int VIEW_MODE_FULL = 7;

    public static int viewMode = VIEW_MODE_RGBA;

    private MenuItem mItemPreviewRGBA;
    private MenuItem mItemPreviewColor;
    private MenuItem mItemPreviewShape;
    private MenuItem mItemPreviewSign;
    private MenuItem mItemPreviewRect;
    private MenuItem mItemPreviewLane;
    private MenuItem mItemPreviewNet;
    private MenuItem mItemPreviewFull;


    public static final int MY_PERMISSIONS_REQUEST_CAMERA = 0;

    private static final boolean FIXED_FRAME_SIZE = true;
    private static final int FRAME_SIZE_WIDTH = 1280;
    private static final int FRAME_SIZE_HEIGHT = 720;

    private static final int LANES_FRAME_SIZE_WIDTH = 720;
    private static final int LANES_FRAME_SIZE_HEIGHT = 960;
    private static final float  LANES_SENSOR_WIDTH_SCALE = 1.5f;

    public static final int NUM_TEMPLATES = 43;


    private CameraBridgeViewBase mOpenCvCameraView;

    private SensorManager mSensorManager;
    private Sensor mRotationSensor;

    float[] accelerometerReading;
    float[] magnetometerReading;
    private double[] intrinsic = {973.939697265625, 0, 639.5, 0, 973.939697265625, 359.5, 0, 0, 1};
    private float[] distortion = {0.1294488310813904f, -0.6889675855636597f, 0, 0, 0.9193099141120911f};
    private float mSensorFocalLength;
    private float mSensorHorizontalAngle;
    private float mSensorVerticalAngle;
    private float mSensorWidth;
    private float mSensorHeight;

    private float[] mRotationMatrix;
    private float[] mOrientationAngles;

    private long mLastMeasuredTime;

    private Boolean cameraPermissionGranted = false;

    // image after color filtering
    private Mat bwRed;
    private Mat bwBlue;
    private Mat bwBlack;
    private Mat bwWhite;
    private Mat bwYellow;
    private Mat bwGreen;
    private Mat bwCyan;
    private Mat bwMagenta;

    private Mat Red;
    private Mat Blue;
    private Mat Black;
    private Mat Gray;
    private Mat White;
    private Mat Yellow;
    private Mat Green;
    private Mat Cyan;
    private Mat Magenta;

    private Mat mInputFrame;
    // black ans white road image
    Mat mRoadImg;
    // image converted to HSV
    private Mat hsv;
    // the downscaled image (for removing noise)
    private static Mat downscaled;
    // the upscaled image (for removing noise)
    private static Mat upscaled;
    // Output image
    private Mat dst;

    // class names for object detection
    private static final String[] classNames = {"background", "speed limit 20", "speed limit 30",
            "speed limit 50", "speed limit 60", "speed limit 70", "speed limit 80",
            "restriction ends 80", "speed limit 100", "speed limit 120", "no overtaking",
            "no overtaking trucks", "priority at next intersection", "priority road", "give way",
            "stop", "no traffic both ways", "no trucks", "no entry", "danger", "bend left",
            "bend right", "bend", "uneven road", "slippery road", "road narrows", "construction",
            "traffic signal", "pedestrian crossing", "school crossing", "cycles crossing", "snow",
            "animals", "restriction ends", "go rights", "go left", "go straight",
            "go right or straight", "go left or straight", "keep right", "keep left", "roundabout",
            "restriction ends overtaking", "restriction ends overtaking trucks"};

    ArrayList<String> mSignNames;

    // object for performing sign detections
    private SignDetection mSignDetection;

    // object for performing lane tracking
    private LaneTracking mLaneTracking;

    // Threads
    private class mSignDetectionThread extends Thread {
        @Override
        public void run() {
            // Sign Detection
            Mat half = dst.submat(0, dst.rows() / 2,
                    dst.cols() / 2, dst.cols());
            mSignDetection.signDetectionPipeline(half, half,
                    SignDetection.OPTION_NETWORK, mSignNames);
        }
    }

    private class mLaneTrackingThread extends Thread {
        @Override
        public void run() {
            // Lane Tracking
            mLaneTracking.mPitchAngle = -mOrientationAngles[2];
            mLaneTracking.LaneTrackingPipeline(mInputFrame, mRoadImg);
        }
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    mRotationMatrix = new float[9];
                    mOrientationAngles = new float[3];

                    bwRed = new Mat();
                    bwBlue = new Mat();
                    bwBlack = new Mat();
                    bwWhite = new Mat();
                    bwYellow = new Mat();
                    bwGreen = new Mat();
                    bwMagenta = new Mat();
                    bwCyan = new Mat();
                    hsv = new Mat();
                    downscaled = new Mat();
                    upscaled = new Mat();

                    mInputFrame = new Mat();
                    dst = new Mat();

                    Red = new Mat();
                    Blue = new Mat();
                    Black = new Mat();
                    Gray = new Mat();
                    White = new Mat();
                    Yellow = new Mat();
                    Green = new Mat();
                    Magenta = new Mat();
                    Cyan = new Mat();

                    mRoadImg = new Mat();

                    mOpenCvCameraView.enableView();

                    mSignDetection = new SignDetection();
                    mLaneTracking = new LaneTracking();

                    mSignNames = new ArrayList<>();

                    // initiate templates for template matching
                    initiateTemplates();

                    // load object detection network model
                    loadNetModel();

                    // load sensor info
                    mLaneTracking.setSensorInfo(mSensorFocalLength, mSensorWidth, mSensorHeight,
                            FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT, LANES_FRAME_SIZE_WIDTH,
                            LANES_FRAME_SIZE_HEIGHT, LANES_SENSOR_WIDTH_SCALE);

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    // Accelerometer sensor listener
    SensorEventListener selAcc = new SensorEventListener() {
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
        }

        public void onSensorChanged(SensorEvent event) {
            accelerometerReading = event.values;
        }
    };

    // Magnometer sensor listener
    SensorEventListener selMag = new SensorEventListener() {
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
        }

        public void onSensorChanged(SensorEvent event) {
            magnetometerReading = event.values;
        }
    };


    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.autonomous_phone_car_surface_view);

        Log.d("verify openCV", String.valueOf(OpenCVLoader.initDebug()));

        // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            // Permission is not granted
            if (!cameraPermissionGranted) {
                // Should we show an explanation?
                if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                        Manifest.permission.CAMERA)) {
                    // Show an explanation to the user *asynchronously* -- don't block
                    // this thread waiting for the user's response! After the user
                    // sees the explanation, try again to request the permission.
                } else {
                    // No explanation needed; request the permission
                    ActivityCompat.requestPermissions(this,
                            new String[]{Manifest.permission.CAMERA},
                            MY_PERMISSIONS_REQUEST_CAMERA);

                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    // MY_PERMISSIONS_REQUEST_READ_CAMERA is an
                    // app-defined int constant. The callback method gets the
                    // result of the request.
                }
            }
        } else {
            // Permission has already been granted
            initiateSensors();
        }

    }


    /**
     * Called when permission requested
     */
    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_CAMERA: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                    cameraPermissionGranted = true;
                    initiateSensors();
                } else {
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.

                    // ask again!
                    ActivityCompat.requestPermissions(this,
                            new String[]{Manifest.permission.CAMERA},
                            MY_PERMISSIONS_REQUEST_CAMERA);
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request.
        }
    }


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
//        if (!OpenCVLoader.initDebug()) {
//            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
////            System.loadLibrary("opencv_java4");
////            try {
////                wait(500);
////                onResume();
////            } catch (InterruptedException e) {
////                e.printStackTrace();
////            }
//        } else {
        OpenCVLoader.initDebug();
        Log.d(TAG, "OpenCV library found inside package. Using it!");
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewColor = menu.add("Color");
        mItemPreviewShape = menu.add("Shape");
        mItemPreviewSign = menu.add("Sign");
        mItemPreviewRect = menu.add("Rectification");
        mItemPreviewLane = menu.add("Lane");
        mItemPreviewNet = menu.add("Net");
        mItemPreviewFull = menu.add("Full");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        else if (item == mItemPreviewColor)
            viewMode = VIEW_MODE_COLOR;
        else if (item == mItemPreviewShape)
            viewMode = VIEW_MODE_SHAPE;
        else if (item == mItemPreviewSign)
            viewMode = VIEW_MODE_SIGN;
        else if (item == mItemPreviewRect)
            viewMode = VIEW_MODE_RECT;
        else if (item == mItemPreviewLane)
            viewMode = VIEW_MODE_LANE;
        else if (item == mItemPreviewNet)
            viewMode = VIEW_MODE_NET;
        else if (item == mItemPreviewFull)
            viewMode = VIEW_MODE_FULL;
        return true;
    }


    public void onCameraViewStarted(int width, int height) {
    }


    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();

        int rows = rgba.rows();
        int cols = rgba.cols();

        // Rotation matrix based on current readings from accelerometer and magnetometer.
        SensorManager.getRotationMatrix(mRotationMatrix, null,
                accelerometerReading, magnetometerReading);
        // Express the updated rotation matrix as three orientation angles.
        SensorManager.getOrientation(mRotationMatrix, mOrientationAngles);

        // labels variables
        int fontFace;
        double scale;
        int thickness;
        int[] baseline = new int[1];
        Size text;


        switch (MainActivity.viewMode) {

            case VIEW_MODE_RGBA:
                dst = inputFrame.rgba();
                break;

            case VIEW_MODE_COLOR:

                // warning: severe memory leak

                // the image to output on the screen in the end
                // -> get the unchanged color image
                Imgproc.cvtColor(inputFrame.rgba(), mInputFrame, Imgproc.COLOR_RGBA2RGB);

                // down-scale and upscale the image to filter out the noise
                Imgproc.pyrDown(mInputFrame, downscaled, new Size(cols / 2, rows / 2));

//                // color filtering with hsv space
//
//                // convert the image from RGBA to HSV
//                Imgproc.cvtColor(upscaled, hsv, Imgproc.COLOR_RGB2HSV);
//
//                //Red color filter
////                ImageOperations.colorFilter(hsv, bwRed, ImageOperations.ColorName.RED);
//                ImageOperations.colorFilter(upscaled, bwRed, ImageOperations.ColorName.RED_ADAPTIVE);
//
//                //Blue color filter
////                ImageOperations.colorFilter(hsv, bwBlue, ImageOperations.ColorName.BLUE);
//                ImageOperations.colorFilter(upscaled, bwBlue, ImageOperations.ColorName.BLUE_ADAPTIVE);
//
//                //Black color filter
////                ImageOperations.colorFilter(hsv, bwBlack, ImageOperations.ColorName.BLACK);
//                ImageOperations.colorFilter(upscaled, bwBlack, ImageOperations.ColorName.ACHROMATIC);
//
//                ImageOperations.colorFilter(hsv, bwWhite, ImageOperations.ColorName.WHITE);
//
//                ImageOperations.colorFilter(hsv, bwYellow, ImageOperations.ColorName.YELLOW);
//
//                ImageOperations.colorFilter(hsv, bwGreen, ImageOperations.ColorName.GREEN);
//
//                Imgproc.dilate(bwRed, bwRed, new Mat(), new Point(-1, 1), 1);
//
//                Imgproc.cvtColor(mInputFrame, dst, Imgproc.COLOR_RGBA2RGB);
//
//                dst.setTo(new Scalar(128, 128, 128));
//                dst.setTo(new Scalar(255, 255, 255), bwWhite);
//                dst.setTo(new Scalar(255, 255, 0), bwYellow);
//                dst.setTo(new Scalar(0, 255, 0), bwGreen);
//                dst.setTo(new Scalar(0, 0, 0), bwBlack);
//                dst.setTo(new Scalar(255, 0, 0), bwRed);
//                dst.setTo(new Scalar(0, 0, 255), bwBlue);
//
//                dst.convertTo(dst, mInputFrame.type());
//
//                break;
                // color filtering with Ostu's method
                List<Mat> rgbChannels = new ArrayList<>();
                Mat floatImage = new Mat();
                downscaled.convertTo(floatImage, CvType.CV_32FC3);
                // get the rgb channels
                Core.split(floatImage, rgbChannels);

                Mat channelsSum = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Mat normalizedRed = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Mat normalizedBlue = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Mat normalizedGreen = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Mat normalizedAchromatic = new Mat(rows/2, cols/2, CvType.CV_8U);
                double threshR, threshG, threshB, threshAch, threshW, threshY, threshM, threshC;
                // sum of channels
                Core.add(rgbChannels.get(0), rgbChannels.get(1), channelsSum);
                Core.add(channelsSum, rgbChannels.get(2), channelsSum);
                // normalized colors
                Core.divide(rgbChannels.get(0), channelsSum, normalizedRed);
                Core.divide(rgbChannels.get(1), channelsSum, normalizedGreen);
                Core.divide(rgbChannels.get(2), channelsSum, normalizedBlue);
                // convert to bytes
                Core.multiply(normalizedRed, new Scalar(255), normalizedRed);
                Core.multiply(normalizedGreen, new Scalar(255), normalizedGreen);
                Core.multiply(normalizedBlue, new Scalar(255), normalizedBlue);
                normalizedRed.convertTo(Red, CvType.CV_8U);
                normalizedGreen.convertTo(Green, CvType.CV_8U);
                normalizedBlue.convertTo(Blue, CvType.CV_8U);
                // normalized yellow
                Mat normalizedYellow = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Core.add(normalizedRed, normalizedGreen, normalizedYellow);
                Core.multiply(normalizedYellow, new Scalar(0.5), normalizedYellow);
                normalizedYellow.convertTo(Yellow, CvType.CV_8U);
                // normalized magenta
                Mat normalizedMagenta = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Core.add(normalizedRed, normalizedBlue, normalizedMagenta);
                Core.multiply(normalizedMagenta, new Scalar(0.5), normalizedMagenta);
                normalizedMagenta.convertTo(Magenta, CvType.CV_8U);
                // normalized cyan
                Mat normalizedCyan = new Mat(rows/2, cols/2, CvType.CV_32FC1);
                Core.add(normalizedBlue, normalizedGreen, normalizedCyan);
                Core.multiply(normalizedCyan, new Scalar(0.5), normalizedCyan);
                normalizedCyan.convertTo(Cyan, CvType.CV_8U);
                // achromatic
                Core.min(Red, Green, Gray);
                Core.min(Gray, Blue, Gray);
                Core.multiply(Gray, new Scalar(3), Gray);
                // Otsu's Thresholds
                threshR = Imgproc.threshold(Red, bwRed, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                threshG = Imgproc.threshold(Green, bwGreen, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                threshB = Imgproc.threshold(Blue, bwBlue, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                threshAch = Imgproc.threshold(Gray, bwBlack, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                threshY = Imgproc.threshold(Yellow, bwYellow, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                threshM = Imgproc.threshold(Magenta, bwMagenta, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                threshC = Imgproc.threshold(Cyan, bwCyan, 0, 255,
                        Imgproc.THRESH_OTSU | Imgproc.THRESH_BINARY);
                // black and whites threshold

                dst.create(downscaled.size(), CvType.CV_8UC3);
                dst.setTo(new Scalar(0));
                double[] thresholds = {threshR, threshG, threshB, threshAch, threshY, threshM, threshC};
                Arrays.sort(thresholds);

                for (int i = 0; i < 7; i++){
                    if (thresholds[i] == threshR)
                        dst.setTo(new Scalar(255,0,0), bwRed);
                    else if (thresholds[i] == threshG)
                        dst.setTo(new Scalar(0,255,0), bwGreen);
                    else if (thresholds[i] == threshB)
                        dst.setTo(new Scalar(0,0,255), bwBlue);
                    else if (thresholds[i] == threshAch)
                        dst.setTo(new Scalar(128, 128, 128), bwBlack);
                    else if (thresholds[i] == threshY)
                        dst.setTo(new Scalar(255,255,0), bwYellow);
                    else if (thresholds[i] == threshM)
                        dst.setTo(new Scalar(255,0,255), bwMagenta);
                    else if (thresholds[i] == threshC)
                        dst.setTo(new Scalar(0, 255, 255), bwCyan);
                }


                // release
                floatImage.release();
                normalizedRed.release();
                normalizedBlue.release();
                normalizedGreen.release();
                normalizedAchromatic.release();
                normalizedYellow.release();
                normalizedMagenta.release();
                normalizedCyan.release();

                Imgproc.pyrUp(dst, dst, new Size(cols, rows));

                break;


            case VIEW_MODE_SHAPE:
                Imgproc.cvtColor(rgba, dst, Imgproc.COLOR_RGBA2RGB);

                List<ImageOperations.Shape> shapes = new ArrayList<>(); // detected shapes lists
                mSignDetection.shapeDetectionPipeline(dst, ImageOperations.ColorName.RED_ADAPTIVE,
                        dst, shapes, true);
                mSignDetection.shapeDetectionPipeline(dst, ImageOperations.ColorName.BLUE_ADAPTIVE,
                        dst, shapes, true);
                mSignDetection.shapeDetectionPipeline(dst, ImageOperations.ColorName.ACHROMATIC,
                        dst, shapes, true);

                break;

            case VIEW_MODE_SIGN:
                Imgproc.cvtColor(rgba, dst, Imgproc.COLOR_RGBA2RGB);
                mInputFrame = dst.submat(0, rows / 2, 0, cols);
                mSignDetection.signDetectionPipeline(mInputFrame, mInputFrame,
                        SignDetection.OPTION_CLASSIC_IM_PROC, mSignNames);

                break;

            case VIEW_MODE_RECT:
                Imgproc.cvtColor(rgba, mInputFrame, Imgproc.COLOR_RGBA2RGB);
                float roll = -mOrientationAngles[2];

                mLaneTracking.mPitchAngle = roll;
                mLaneTracking.birdsEyeTransform(mInputFrame, mRoadImg, new Mat());

                double frameScale = rows / (double) mRoadImg.height();
                int scaledWidth = (int) Math.round(frameScale * mRoadImg.width());
                int scaledHeight = (int) Math.round(frameScale * mRoadImg.height());
                int offset = cols / 2 - (scaledWidth/2);

                Imgproc.resize(mRoadImg, mRoadImg, new Size(scaledWidth, scaledHeight));

                dst.create(mInputFrame.size(), mInputFrame.type());
                dst.setTo(new Scalar(0));

                mRoadImg.copyTo(dst.colRange(offset, offset + scaledWidth));

                String eulerLabel = String.format("roll angle: %.1f", Math.toDegrees(roll));
                fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
                scale = 2;//0.4;
                thickness = 2;//1;
                baseline = new int[1];
                text = Imgproc.getTextSize(eulerLabel, fontFace, scale, thickness, baseline);

                Imgproc.putText(dst, eulerLabel, new Point(100, 300), fontFace, scale,
                        ImageOperations.RGB_RED, thickness);

                break;

            case VIEW_MODE_LANE:

                Imgproc.cvtColor(rgba, mInputFrame, Imgproc.COLOR_RGBA2RGB);
                mSignNames.add("go left");

                mLaneTracking.mPitchAngle = -mOrientationAngles[2];
                mLaneTracking.LaneTrackingPipeline(mInputFrame, dst);

                mLaneTracking.calculateDriving(dst, dst, mSignNames);

                mLaneTracking.drawLanes(mInputFrame, dst, dst);

                break;

            case VIEW_MODE_NET:
                Imgproc.cvtColor(rgba, dst, Imgproc.COLOR_RGBA2RGB);
//                mInputFrame = dst.submat(0, rows/2, 0, cols/2);
//                mSignDetection.signDetectionPipeline(mInputFrame, mInputFrame,
//                        SignDetection.OPTION_NETWORK, mSignNames);
                mInputFrame = dst.submat(0, rows / 2, cols / 2, cols);
                mSignDetection.signDetectionPipeline(dst, dst,
                        SignDetection.OPTION_NETWORK, mSignNames);

                break;

            case VIEW_MODE_FULL:

                Imgproc.cvtColor(rgba, mInputFrame, Imgproc.COLOR_RGBA2RGB);
                dst = mInputFrame.clone();

//                // One thread process
//
//                // Sign Detection
//                Mat half = dst.submat(0, rows / 2, cols / 2, cols);
//                mSignDetection.signDetectionPipeline(half, half,
//                        SignDetection.OPTION_NETWORK, mSignNames);
//
//                // Lane Tracking
//                mLaneTracking.mPitchAngle = -mOrientationAngles[2];
//                mLaneTracking.LaneTrackingPipeline(mInputFrame, mRoadImg);

                // Two threads process

                mSignDetectionThread signThread = new mSignDetectionThread();
                mLaneTrackingThread laneThread = new mLaneTrackingThread();

                signThread.start();
                laneThread.start();

                try {
                    signThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                try {
                    laneThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                mLaneTracking.calculateDriving(mRoadImg, mRoadImg, mSignNames);

                mLaneTracking.drawLanes(mInputFrame, mRoadImg, dst);

                break;

        }

        // display FPS
        double fps = 1.0 / ((SystemClock.elapsedRealtime() - mLastMeasuredTime) / 1000.0);
        String fpsLabel = String.format("FPS: %.2f", fps);
        fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
        scale = 0.4;
        thickness = 1;
        baseline = new int[1];
        text = Imgproc.getTextSize(fpsLabel, fontFace, scale, thickness, baseline);

        Imgproc.putText(dst, fpsLabel, new Point(20, 20), fontFace, scale,
                ImageOperations.RGB_RED, thickness);

        mLastMeasuredTime = SystemClock.elapsedRealtime();

        return dst;
    }


    // Set camera and sensors listeners
    public void initiateSensors() {

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.autonomous_phone_car_surface_view);
        if (FIXED_FRAME_SIZE) {
            mOpenCvCameraView.setMaxFrameSize(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT);
        }
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mOpenCvCameraView.setCameraPermissionGranted();


//         Get camera characteristics
        CameraManager mCameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

        try {
            Log.d(TAG, "tryAcquire");
            Semaphore mCameraOpenCloseLock = new Semaphore(1);
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
            String cameraId = sharedPreferences.getString("prefCamera", "0");

            // Choose the sizes for camera preview and video recording
            CameraCharacteristics characteristics = mCameraManager.getCameraCharacteristics(cameraId);

            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            int cOrientation = characteristics.get(CameraCharacteristics.LENS_FACING);
            if (cOrientation == CameraCharacteristics.LENS_FACING_BACK) {
                float[] maxFocus = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                mSensorFocalLength = maxFocus[0];
                SizeF size = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                mSensorWidth = size.getWidth();
                mSensorHeight = size.getHeight();
                mSensorHorizontalAngle = (float) (2 * Math.atan(mSensorWidth / (mSensorFocalLength * 2)));
                mSensorVerticalAngle = (float) (2 * Math.atan(mSensorHeight / (mSensorFocalLength * 2)));
            }

//            // not supported in our phones
//            int[] capabilities;
////            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
////                capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
//////                boolean supportDepth = arrayContains(capabilities,
//////                        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT);
////                        }
////            if(Build.VERSION.SDK_INT >= 23) {
////                intrinsic = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
////                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
////                    distortion = characteristics.get(CameraCharacteristics.LENS_DISTORTION);
////                }
////                else {
////                    distortion = characteristics.get(CameraCharacteristics.LENS_RADIAL_DISTORTION);
////                }
////            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
            Toast.makeText(this, "Cannot access the camera.", Toast.LENGTH_SHORT).show();
            this.finish();
        } catch (SecurityException e) {
            e.printStackTrace();
            Toast.makeText(this, "Cannot access the camera.", Toast.LENGTH_SHORT).show();
            this.finish();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        // Connect Rotation sensor listener
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mRotationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);

        List<Sensor> list = mSensorManager.getSensorList(Sensor.TYPE_ACCELEROMETER);
        if (list.size() > 0) {
            mSensorManager.registerListener(selAcc, (Sensor) list.get(0), SensorManager.SENSOR_DELAY_NORMAL);
        } else {
            Toast.makeText(getBaseContext(), "Error: No Accelerometer.", Toast.LENGTH_LONG).show();
        }

        list = mSensorManager.getSensorList(Sensor.TYPE_MAGNETIC_FIELD);
        if (list.size() > 0) {
            mSensorManager.registerListener(selMag, (Sensor) list.get(0), SensorManager.SENSOR_DELAY_NORMAL);
        } else {
            Toast.makeText(getBaseContext(), "Error: No Magnetic Field Sensor.", Toast.LENGTH_LONG).show();
        }
    }


    // initiate templates for template matching
    private void initiateTemplates() {

        mSignDetection.mTemplates = new ArrayList<>(NUM_TEMPLATES);

        for (int i = 0; i < NUM_TEMPLATES; i++) {

            // load the specified image from file system in bgra color

            mSignDetection.mTemplates.add(new TemplateMatching.SignTemplate());

            mSignDetection.mTemplates.get(i).id = i;
            // load image
            String imageString = String.format("Meta/%d.png", i);
            try {
                InputStream is = getAssets().open(imageString);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                Utils.bitmapToMat(bitmap, mSignDetection.mTemplates.get(i).img);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // determine traffic sign type
            if ((i < 10 && i != 6) || i == 15 || i == 16)
                mSignDetection.mTemplates.get(i).type = TemplateMatching.SignType.PROHIBITORY;
            else if (i == 11 || (i >= 18 && i <= 31))
                mSignDetection.mTemplates.get(i).type = TemplateMatching.SignType.DANGER;
            else if (i >= 33 && i <= 40)
                mSignDetection.mTemplates.get(i).type = TemplateMatching.SignType.MANDATORY;
            else
                mSignDetection.mTemplates.get(i).type = TemplateMatching.SignType.OTHER;

            // make black and white image for template
            TemplateMatching.initiateTemplate(mSignDetection.mTemplates.get(i));
        }
    }


    // initiate model for object detection
    private void loadNetModel() {

        String model = getPath("object_detection_network/frozen_inference_graph.pb", this);
        String config = getPath("object_detection_network/graph_config.pbtxt", this);
        if (mSignDetection.setNetwork(model, config, classNames))
            Log.i(TAG, "Network loaded successfully");
        else
            Log.i(TAG, "Network Failed to be loaded");
    }


    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File tmp = new File(file);
            File outFile = new File(context.getFilesDir(), tmp.getName());
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }
}
