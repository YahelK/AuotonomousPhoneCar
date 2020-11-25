package com.autonomousphonecar;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class SignDetection {

    // sign detection options constant
    final static int OPTION_CLASSIC_IM_PROC = 0;
    final static int OPTION_NETWORK = 1;
    final static int OPTION_COMBINED = 2; //TODO: add combined method

    // object detection constants
    final static int OBJECT_NET_IN_WIDTH = 320;
    final static int OBJECT_NET_IN_HEIGHT = 320;
    final static float OBJECT_NET_WH_RATIO = (float) OBJECT_NET_IN_WIDTH / OBJECT_NET_IN_HEIGHT;
    final static float OBJECT_NET_IN_SCALE_FACTOR = 1.0f;
    final static float OBJECT_NET_MEAN_VAL = 0f;
    final static float OBJECT_NET_THRESHOLD = 0.5f;

    // the Network and it's labels
    private boolean ismNetLoaded = false;
    private Net mNet;
    private String[] mClassNames = {"background", "speed limit 20", "speed limit 30",
            "speed limit 50", "speed limit 60", "speed limit 70", "speed limit 80",
            "restriction ends 80", "speed limit 100", "speed limit 120", "no overtaking",
            "no overtaking trucks", "priority at next intersection", "priority road", "give way",
            "stop", "no traffic both ways", "no trucks", "no entry", "danger", "bend left",
            "bend right", "bend", "uneven road", "slippery road", "road narrows", "construction",
            "traffic signal", "pedestrian crossing", "school crossing", "cycles crossing", "snow",
            "animals", "restriction ends", "go rights", "go left", "go straight",
            "go right or straight", "go left or straight", "keep right", "keep left", "roundabout",
            "restriction ends overtaking", "restriction ends overtaking trucks"};


    // class intermediates mats
    private Mat downscaled = new Mat();
    private Mat bw = new Mat();
    private Mat sobelx = new Mat();
    private Mat sobely = new Mat();

    // templates for template matching
    public static List<TemplateMatching.SignTemplate> mTemplates;


    /**
     * Main function, get an image and find traffic signs.
     *
     * @param src       input image in rgba format.
     * @param dst       output image with detected signs marked
     * @param option    set the detection method.
     * @param signNames output list of founded sign names
     */
    public void signDetectionPipeline(Mat src, Mat dst, int option, ArrayList<String> signNames) {
        int rows = src.rows();
        int cols = src.cols();

        switch (option) {

            case OPTION_CLASSIC_IM_PROC:
                Imgproc.pyrDown(dst, downscaled, new Size(cols / 2, rows / 2));

                List<ImageOperations.Shape> shapes = new ArrayList<>(); // detected shapes lists
                shapeDetectionPipeline(downscaled, ImageOperations.ColorName.RED_ADAPTIVE, dst, shapes, false);
                shapeDetectionPipeline(downscaled, ImageOperations.ColorName.BLUE_ADAPTIVE, dst, shapes, false);
                shapeDetectionPipeline(downscaled, ImageOperations.ColorName.ACHROMATIC, dst, shapes, false);

                for (ImageOperations.Shape shape : shapes)
                    TemplateMatching.signMatching(dst, downscaled, shape, mTemplates,
                            TemplateMatching.ColorMode.BW);

                break;

            case OPTION_NETWORK:
                ObjectDetectionPipeline(dst, dst, signNames);
        }
    }

    /**
     * Find shapes in specific color
     *
     * @param src       input image in rgb format.
     * @param color     color filter method (RED_ADAPTIVE, BLUE_ADAPTIVE, ACHROMATIC)
     * @param dst       output image with detected shapes labeled
     * @param shapes    list to keep the shapes
     * @param shapeMode display founded shapes names
     */
    public void shapeDetectionPipeline(Mat src, ImageOperations.ColorName color, Mat dst,
                                        List<ImageOperations.Shape> shapes, boolean shapeMode) {

        ImageOperations.colorFilter(src, bw, color);
        Imgproc.Sobel(bw, sobelx, -1, 1, 0);
        Imgproc.Sobel(bw, sobely, -1, 1, 0);
        Core.bitwise_or(sobelx, sobely, bw);
        Imgproc.dilate(bw, bw, new Mat(), new Point(-1, -1), 1); // TODO: check if dilating is necessary

        switch (color) {
            case RED_ADAPTIVE:
            case RED:
                ImageOperations.shapeDetection(bw, dst, shapes, ImageOperations.ColorName.RED, shapeMode,
                        ImageOperations.ShapeName.TRIANGLE, ImageOperations.ShapeName.OCTAGON,
                        ImageOperations.ShapeName.CIRCLE);
                break;
            case BLUE_ADAPTIVE:
            case BLUE:
                ImageOperations.shapeDetection(bw, dst, shapes, ImageOperations.ColorName.BLUE, shapeMode,
                        ImageOperations.ShapeName.CIRCLE);
                break;
            case ACHROMATIC:
            case BLACK:
                ImageOperations.shapeDetection(bw, dst, shapes, ImageOperations.ColorName.BLACK, shapeMode,
                        ImageOperations.ShapeName.SQUARE, ImageOperations.ShapeName.CIRCLE);
                break;
            default:
                break;
        }
    }


    /**
     * Find objects with object detection network (needed to set network first)
     *
     * @param src        input image in rgb format.
     * @param dst        output image with bounded objects
     * @param classNames output list of founded classes names
     */
    private void ObjectDetectionPipeline(Mat src, Mat dst, ArrayList<String> classNames) {
        if (src != dst) {
            src.copyTo(dst);
        }
        int cols = dst.cols();
        int rows = dst.rows();


        // Forward image through the Network.
        Mat blob = Dnn.blobFromImage(dst, OBJECT_NET_IN_SCALE_FACTOR,
                new Size(OBJECT_NET_IN_WIDTH, OBJECT_NET_IN_HEIGHT),
                new Scalar(OBJECT_NET_MEAN_VAL, OBJECT_NET_MEAN_VAL, OBJECT_NET_MEAN_VAL),
                false, false, CvType.CV_8U);
        mNet.setInput(blob);
        Mat detections = mNet.forward();
        detections = detections.reshape(1, (int) detections.total() / 7);

        for (int i = 0; i < detections.rows(); ++i) {

            double confidence = detections.get(i, 2)[0];

            if (confidence > OBJECT_NET_THRESHOLD) {
                int classId = (int) detections.get(i, 1)[0];
                int left = (int) (detections.get(i, 3)[0] * cols);
                int top = (int) (detections.get(i, 4)[0] * rows);
                int right = (int) (detections.get(i, 5)[0] * cols);
                int bottom = (int) (detections.get(i, 6)[0] * rows);

                classNames.add(mClassNames[classId]);

                // Draw rectangle around detected object.
                Imgproc.rectangle(dst, new Point(left, top), new Point(right, bottom),
                        new Scalar(0, 255, 0));
                String label = mClassNames[classId] + ": " + confidence;
                int[] baseLine = new int[1];
                Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.5, 1, baseLine);

                // Draw background for label.
                Imgproc.rectangle(dst, new Point(left, top - labelSize.height),
                        new Point(left + labelSize.width, top + baseLine[0]),
                        new Scalar(255, 255, 255), Imgproc.FILLED);

                // Write class name and confidence.
                Imgproc.putText(dst, label, new Point(left, top),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));
            }
        }
    }

    /**
     * Set the class network and class labels list
     *
     * @param modelPath  string of the full path to tensorFlow frozen graph .pb
     * @param configPath string of the full path to config of the frozen graph .pbtxt
     * @param classNames list of strings of the model classes
     */
    public boolean setNetwork(String modelPath, String configPath, String[] classNames) {
        if (modelPath != null && mClassNames != null) {
            mClassNames = classNames;
            mNet = Dnn.readNetFromTensorflow(modelPath, configPath);
            if (!mNet.empty()) {
                ismNetLoaded = true;
                return true;
            }
        }
        return false;
    }

}
