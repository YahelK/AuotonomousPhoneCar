package com.auotonomousphonecar;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
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
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";

    public static final int VIEW_MODE_RGBA = 0;
    public static final int VIEW_MODE_COLOR = 1;
    public static final int VIEW_MODE_SHAPE = 2;
    public static final int VIEW_MODE_SIGN = 3;
    public static final int VIEW_MODE_RECT = 4;
    public static final int VIEW_MODE_LANE = 5;

    public static final int MY_PERMISSIONS_REQUEST_CAMERA = 0;


    private MenuItem mItemPreviewRGBA;
    private MenuItem mItemPreviewColor;
    private MenuItem mItemPreviewShape;
    private MenuItem mItemPreviewSign;
    private MenuItem mItemPreviewRect;
    private MenuItem mItemPreviewLane;

    private CameraBridgeViewBase mOpenCvCameraView;

    private SensorManager mSensorManager;
    private Sensor mRotationSensor;
    //
//    float[] rotationValues = {0, 0, 0};
    float[] accelerometerReading;
    float[] magnetometerReading;

    private static final boolean FIXED_FRAME_SIZE = true;
    private static final int FRAME_SIZE_WIDTH = 1280;
    private static final int FRAME_SIZE_HEIGHT = 720;

    public static final int NUM_TEMPLATES = 43;

    private Mat mIntermediateMat;

    public static int viewMode = VIEW_MODE_RGBA;

    private Boolean cameraPermissionGranted= false;

    // image after color filtering
    private Mat bwRed;
    private Mat bwBlue;
    private Mat bwBlack;
    private Mat bwWhite;
    private Mat bwYellow;
    private Mat bwGreen;

    private Mat Red;
    private Mat Blue;
    private Mat Black;
    private Mat Gray;
    private Mat White;
    private Mat Yellow;
    private Mat Green;


    private Mat mInputFrame;
    // image converted to HSV
    private Mat hsv;
    // the downscaled image (for removing noise)
    private static Mat downscaled;
    // the upscaled image (for removing noise)
    private static Mat upscaled;
    // detected shapes lists
    private static List<ImageOperations.Shape> shapes;
    // traffic signs templates
    private static List<TemplateMatching.SignTemplate> templates;
    // RGBA channels
    private static List<Mat> channels;
    // Output image
    private Mat dst;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    bwRed = new Mat();
                    bwBlue = new Mat();
                    bwBlack = new Mat();
                    bwWhite = new Mat();
                    bwYellow = new Mat();
                    bwGreen = new Mat();
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

                    channels = new ArrayList<>();

                    mOpenCvCameraView.enableView();

                    // initiate templates for template matching
                    initiateTemplates();

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() { Log.i(TAG, "Instantiated new " + this.getClass()); }

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

    public void initiateSensors() {

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.autonomous_phone_car_surface_view);
        if (FIXED_FRAME_SIZE) {
            mOpenCvCameraView.setMaxFrameSize(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT);
        }
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

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
    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.autonomous_phone_car_surface_view);

        Log.d("verify openCV",String.valueOf(OpenCVLoader.initDebug()));

        // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            // Permission is not granted
            if(!cameraPermissionGranted) {
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

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
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
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        mIntermediateMat = new Mat();
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();

        mIntermediateMat = null;
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        Size sizeRgba = rgba.size();

        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;


        switch (MainActivity.viewMode) {

            case VIEW_MODE_RGBA:
                break;

            case VIEW_MODE_COLOR:

                // the image to output on the screen in the end
                // -> get the unchanged color image
                mInputFrame = inputFrame.rgba();

                // down-scale and upscale the image to filter out the noise
                Imgproc.pyrDown(mInputFrame, downscaled, new Size(cols / 2, rows / 2));
                Imgproc.pyrUp(downscaled, upscaled, new Size(cols, rows));

                // convert the image from RGBA to HSV
                Imgproc.cvtColor(upscaled, hsv, Imgproc.COLOR_RGB2HSV);

                //Red color filter
                ImageOperations.colorFilter(hsv, bwRed, ImageOperations.ColorName.RED);

                //Blue color filter
                ImageOperations.colorFilter(hsv, bwBlue, ImageOperations.ColorName.BLUE);

                //Black color filter
                ImageOperations.colorFilter(hsv, bwBlack, ImageOperations.ColorName.BLACK);

                ImageOperations.colorFilter(hsv, bwWhite, ImageOperations.ColorName.WHITE);

                ImageOperations.colorFilter(hsv, bwYellow, ImageOperations.ColorName.YELLOW);

                ImageOperations.colorFilter(hsv, bwGreen, ImageOperations.ColorName.GREEN);

                Imgproc.dilate(bwRed, bwRed, new Mat(), new Point(-1, 1), 1);

                Imgproc.cvtColor(mInputFrame, dst, Imgproc.COLOR_RGBA2RGB);

                dst.setTo(new Scalar(128, 128, 128));
                dst.setTo(new Scalar(255, 255, 255), bwWhite);
                dst.setTo(new Scalar(255, 255, 0), bwYellow);
                dst.setTo(new Scalar(0, 255, 0), bwGreen);
                dst.setTo(new Scalar(0, 0, 0), bwBlack);
                dst.setTo(new Scalar(255, 0, 0), bwRed);
                dst.setTo(new Scalar(0, 0, 255), bwBlue);

                dst.convertTo(dst, mInputFrame.type());

                return dst;

            case VIEW_MODE_SHAPE:
                ImageOperations.memoryTest(inputFrame.rgba());
                break;
            case VIEW_MODE_SIGN:

                // the image to output on the screen in the end
                // -> get the unchanged color image
                mInputFrame = inputFrame.rgba();
                Imgproc.cvtColor(mInputFrame, dst, Imgproc.COLOR_RGBA2RGB);

                Red.setTo(new Scalar(0, 0, 0));
                Blue.setTo(new Scalar(0, 0, 0));
                Black.setTo(new Scalar(0, 0, 0));

                Imgproc.pyrDown(dst, downscaled, new Size(cols / 2, rows / 2));

                // convert the image from RGBA to HSV
                Imgproc.cvtColor(downscaled, hsv, Imgproc.COLOR_RGB2HSV);
//                Red color filter
                ImageOperations.colorFilter(downscaled, bwRed, ImageOperations.ColorName.RED_ADAPTIVE);
                // delete small contours
//                deleteSmallContours(bwRed);
                downscaled.copyTo(Red, bwRed);
//                // make edges image
                Imgproc.Canny(Red, bwRed, 100, 200);
                Imgproc.dilate(bwRed, bwRed, new Mat(), new Point (-1,-1), 1);
//                Imgproc.resize(bwRed, bwRed, new Size(cols / 2, rows / 2),
//                        0, 0, Imgproc.INTER_NEAREST);

                //Blue color filter
                ImageOperations.colorFilter(downscaled, bwBlue, ImageOperations.ColorName.BLUE_ADAPTIVE);
                // delete small contours
//                deleteSmallContours(bwBlue);
                downscaled.copyTo(Blue, bwBlue);
                // make edges image
                Imgproc.Canny(Blue, bwBlue, 100, 200);
                Imgproc.dilate(bwBlue, bwBlue, new Mat(), new Point (-1,-1), 1);
//                Imgproc.resize(bwBlue, bwBlue, new Size(cols / 2, rows / 2),
//                        0, 0, Imgproc.INTER_NEAREST);

                //Black color filter
                ImageOperations.colorFilter(hsv, bwBlack, ImageOperations.ColorName.BLACK);
                // delete small contours
//                deleteSmallContours(bwBlack);
                downscaled.copyTo(Black, bwBlack);
                // make edges image
                Imgproc.Canny(Black, bwBlack, 100, 200);
                Imgproc.dilate(bwBlack, bwBlack, new Mat(), new Point (-1,-1), 1);
//                Imgproc.resize(bwBlack, bwBlack, new Size(cols / 2, rows / 2),
//                        0, 0, Imgproc.INTER_NEAREST);


                shapes = new ArrayList<>();

                ImageOperations.shapeDetection(bwRed, dst, shapes, ImageOperations.ColorName.RED,
                        ImageOperations.ShapeName.TRIANGLE, ImageOperations.ShapeName.OCTAGON,
                        ImageOperations.ShapeName.CIRCLE);
                ImageOperations.shapeDetection(bwBlue, dst, shapes, ImageOperations.ColorName.BLUE,
                        ImageOperations.ShapeName.CIRCLE);
                ImageOperations.shapeDetection(bwBlack, dst, shapes, ImageOperations.ColorName.BLACK,
                        ImageOperations.ShapeName.SQUARE, ImageOperations.ShapeName.CIRCLE);

                for (ImageOperations.Shape shape : shapes)
                    TemplateMatching.signMatching(dst, downscaled, shape, templates,
                            TemplateMatching.ColorMode.GRAY);

//                Imgproc.pyrUp(Blue, dst, new Size(cols, rows));
                return dst;

            case VIEW_MODE_RECT:

                // Rotation matrix based on current readings from accelerometer and magnetometer.
                final float[] rotationMatrix = new float[9];
                SensorManager.getRotationMatrix(rotationMatrix, null,
                        accelerometerReading, magnetometerReading);

                // Express the updated rotation matrix as three orientation angles.
                final float[] orientationAngles = new float[3];
                SensorManager.getOrientation(rotationMatrix, orientationAngles);

                float[] values = orientationAngles;

//                float[] values = rotationValues;
//
                Mat rectified = inputFrame.rgba();
//
//                Mat rotationVector = new Mat(3, 1, CvType.CV_32FC1);
//                rotationVector.put(0,0, values);
//                Mat rotMat = setRotationMatrix(-(values[2] + Math.PI / 2.0), "z");
                Mat rotMat = ImageOperations.setRotationMatrix(values[1], "z");
//                rotMat = rotMat.mul(ImageOperations.setRotationMatrix(-(values[2] + Math.PI / 2), "y"));
                Core.gemm(rotMat, ImageOperations.setRotationMatrix((values[2] + (float)(Math.PI / 2.0)), "x"),
                        1, new Mat(), 0, rotMat); // matrix multiplication
//                Core.divide(rotMat, new Scalar(rotMat.get(2, 2)[0]), rotMat);
//                MatOfPoint2f srcPoints = new MatOfPoint2f(), dstPoints = new MatOfPoint2f();
//                List<Point> points = new ArrayList<>();
//                points.add(new Point(0, 0));
//                points.add(new Point(1, 0));
//                points.add(new Point(0, 1));
//                points.add(new Point(1, 1));
//                srcPoints.fromList(points);
//                Core.perspectiveTransform(srcPoints, dstPoints, rotMat);
//                Core.multiply(srcPoints, new Scalar(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT), srcPoints);
//                Core.multiply(dstPoints, new Scalar(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT), dstPoints);
//                rotMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);

//                Core.multiply(rotMat, ImageOperations.setRotationMatrix(0, "x"), rotMat);
//                Calib3d.Rodrigues(rotationVector, rotationMatrix);

                Mat T = new Mat(3, 1, CvType.CV_64F);
                T.put(0, 0, FRAME_SIZE_WIDTH/2);
                T.put(1, 0, FRAME_SIZE_HEIGHT/2);
                T.put(2, 0, 1);
                Core.gemm(rotMat, T, 1, new Mat(), 0, T);

                Mat W =  new Mat(3, 3, CvType.CV_64F);
                W.put(0, 0, 1);
                W.put(1, 1, 1);
                W.put(2, 2, 1);
                W.put(0, 2, FRAME_SIZE_WIDTH/2 - T.get(0, 0)[0]);
                W.put(1, 2, FRAME_SIZE_HEIGHT/2 - T.get(1, 0)[0]);

                Core.gemm(W, rotMat, 1, new Mat(), 0, rotMat);
//
                Mat rotMat2 = new Mat(2, 3, CvType.CV_32FC1);
                rotMat2.put(0, 0, (rotMat.get(0, 0))[0]);
                rotMat2.put(0, 1, (rotMat.get(0, 1)[0]));
                rotMat2.put(1, 0, (rotMat.get(1, 0)[0]));
                rotMat2.put(1, 1, (rotMat.get(1, 1)[0]));
                rotMat2.put(0, 2, (rotMat.get(0, 2)[0]));
                rotMat2.put(1, 2, (rotMat.get(1, 2)[0]));
//
                Size sz = new Size(1280, 720);
//                Imgproc.warpAffine(rectified, rectified, rotMat2, sz);
                Imgproc.warpPerspective(rectified, rectified, rotMat, sz);


                int fontface = Imgproc.FONT_HERSHEY_SIMPLEX;
                double scale = 2;//0.4;
                int thickness = 2;//1;
                int[] baseline = new int[1];

//                String label = String.format("z: %.1f, x: %.1f, y: %.1f", values[0], values[1], values[2]);
                String label = String.format("yawn: %.1f, pitch: %.1f, roll: %.1f",
                        values[1], values[0], values[2] + (float)(Math.PI / 2.0));

                Size text = Imgproc.getTextSize(label, fontface, scale, thickness, baseline);

                Point pt = new Point(
                        100,
                        300
                );

                Imgproc.putText(rectified, label, pt, fontface, scale, ImageOperations.RGB_RED,
                        thickness);


                return rectified;


            case VIEW_MODE_LANE:

                mInputFrame = inputFrame.rgba();
                Mat mask = new Mat();
                Imgproc.cvtColor(mInputFrame, dst, Imgproc.COLOR_RGBA2RGB);

                MatOfPoint2f srcPoints = new MatOfPoint2f(), dstPoints = new MatOfPoint2f();
                List<Point> points = new ArrayList<>();
                points.add(new Point(0, 0));
                points.add(new Point(1, 0));
                points.add(new Point(0, 1));
                points.add(new Point(1, 1));
                dstPoints.fromList(points);
                points.clear();
                points.add(new Point(0.43,0.65));
                points.add(new Point(0.58,0.65));
                points.add(new Point(0.1,1));
                points.add(new Point(1, 1));
                srcPoints.fromList(points);

//                Imgproc.GaussianBlur(mInputFrame, downscaled,
//                        new Size(5, 5),0, 0);
//                Imgproc.medianBlur(frame, dst, 5);

                LaneDetection.pipeline(dst, mask,
                        new Scalar(100, 255), new Scalar(15, 255));
                ImageOperations.perspectiveWarp(mask, mask,
                        new Scalar(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT), srcPoints, dstPoints);
                double steer = LaneDetection.calculateLanes(mask, mask, 9, 150,
                        1, false);
                ImageOperations.perspectiveWarp(mask, mask,
                        new Scalar(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT), dstPoints, srcPoints);
                Core.add(dst, new Scalar(0, 32, 64), dst, mask);

                int fontface0 = Imgproc.FONT_HERSHEY_SIMPLEX;
                double scale0 = 2;// 0.4;
                int thickness0 = 2;// 1;
                String label0 = String.format("steering: %.1f", steer);
                Point pt0 = new Point(100, 300);
                Imgproc.putText(dst, label0, pt0, fontface0, scale0, new Scalar(128, 64, 64), thickness0);

                mask.release();
                srcPoints.release();
                dstPoints.release();

                return dst;

        }



        return rgba;
    }


    // initiate templates for template matching
    private void initiateTemplates() {

        templates = new ArrayList<>(NUM_TEMPLATES);

        for (int i = 0; i < NUM_TEMPLATES; i++) {

            // load the specified image from file system in bgra color

            templates.add(new TemplateMatching.SignTemplate());

            templates.get(i).id = i;
            // load image
            String imageString = String.format("Meta/%d.png", i);
            try {
                InputStream is = getAssets().open(imageString);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                Utils.bitmapToMat(bitmap, templates.get(i).img);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // determine traffic sign type
            if ((i < 10 && i != 6) || i == 15 || i == 16)
                templates.get(i).type = TemplateMatching.SignType.PROHIBITORY;
            else if (i == 11 || (i >= 18 && i <= 31))
                templates.get(i).type = TemplateMatching.SignType.DANGER;
            else if (i >= 33 && i <= 40)
                templates.get(i).type = TemplateMatching.SignType.MANDATORY;
            else
                templates.get(i).type = TemplateMatching.SignType.OTHER;

            // make black and white image for template
            TemplateMatching.initiateTemplate(templates.get(i));
        }
    }


}
