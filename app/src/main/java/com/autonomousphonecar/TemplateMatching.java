package com.autonomousphonecar;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static com.autonomousphonecar.ImageOperations.shapesNames;

public class TemplateMatching {


    public enum SignType {
        PROHIBITORY, DANGER, MANDATORY, OTHER
    }

    public enum ColorMode {
        GRAY, BW, COLORS
    }


    /**
     * Class for traffic signs templates
     * img          Meta sign image
     * bw           black and white image for template matching
     * id           serial number of the traffic sign
     * type         traffic sign type
     * matchResult  Mat for template matching result
     */
    public static class SignTemplate {
        Mat img = new Mat();
        Mat gray = new Mat();
        Mat bw = new Mat();
        Mat mask = new Mat();
        Mat rgb_g_LUTs = new Mat(256, 4, CvType.CV_8U); // red green blue gray LUTs
        int id = -1;
        SignType type = SignType.OTHER;
        Mat matchResult = new Mat();
    }

    /**
     * convert template image to 2 colors image, and initiate matchResult matrix
     *
     * @param template template that is already initiated with id, fd, type and initial bgra img
     */
    public static void initiateTemplate(SignTemplate template) {

        // take the alpha channel
        List<Mat> channels = new ArrayList<>(4);
        Core.split(template.img, channels);
        Mat alphaChannel = channels.get(3);

        // convert RGBA image to BGR
        Imgproc.cvtColor(template.img, template.img, Imgproc.COLOR_RGBA2RGB);


        // convert the image from RGB to HSV
        Imgproc.cvtColor(template.img, template.bw, Imgproc.COLOR_RGB2HSV);
        // filter template colors
        filterByTemplate(template, template.bw);
//        // normalize the template to ones and zeros
//         Core.normalize(template.gray, template.gray, 0, 1, Core.NORM_MINMAX,
//                 -1, new Mat());

        // convert from BGR to gray scale
        Imgproc.cvtColor(template.img, template.gray, Imgproc.COLOR_RGB2GRAY);

        // make mask from alphaChannel
        ImageOperations.colorFilter(alphaChannel, template.mask, ImageOperations.ColorName.ALPHA);

        // resize to a constant size
        ImageOperations.DownSampleImage(template.img);
        ImageOperations.DownSampleImage(template.gray);
        ImageOperations.DownSampleImage(template.bw);
        ImageOperations.DownSampleImage(template.mask);

        // take only not transparent values
        template.gray.copyTo(template.gray, template.mask);
        template.bw.copyTo(template.bw, template.mask);

        // generate LUTs
        Core.split(template.img, channels);
        for (int i = 0; i < 4; i++) {
            Mat hist = new Mat();
            if (i < 3) {
                // calculate cumulative histogram
                ImageOperations.createCumulativeHist(channels.get(i), template.mask, hist);
            } else {
                ImageOperations.createCumulativeHist(template.gray, template.mask, hist);
            }
            // create LUTs of the inverse cumulative histograms functions
            ImageOperations.createInverseMonoLUT(hist, template.rgb_g_LUTs.col(i));
        }

//        // crop the template
//        Rect roi = new Rect(newSizeW / 32, newSizeH / 32,
//                (30 * newSizeW) / 32, (30 * newSizeH) / 32);
//        template.gray = template.gray.submat(roi);
//        template.bw = template.bw.submat(roi);
//
//        // Create the result matrix for template matching
//        // assuming the proposal sign will be newSizeW and newSizeH
//        int result_cols = (4 * newSizeW) / 32;
//        int result_rows = (4 * newSizeH) / 32;
//        template.matchResult.create(result_rows, result_cols, CvType.CV_32FC1);
//        template.matchResult.create(1, 1, CvType.CV_32FC1);

    }


    /**
     * Template matching for given shape and it's black and white image
     *
     * @param dst       output image for labels and rectangles
     * @param org       color image to take from the shape's points
     * @param shape     the shape for template matching
     * @param templates list of templates to matching on
     */
    public static void signMatching(Mat dst, Mat org, ImageOperations.Shape shape, List<SignTemplate> templates,
                                    ColorMode colorMode) {

        double scale = (double)dst.width()/org.width();
        // take the bounding box of the shape from original image
        Rect r = Imgproc.boundingRect(shape.points);
        Mat img = new Mat(org, r);
        List<Mat> channels = new ArrayList<>(3);

        double matchVal = 0; // initiate with big number
        int matchId = 0;
        int match_method = Imgproc.TM_CCOEFF_NORMED;

        // resize and warp to a constant size
        ImageOperations.warpDownSampleImage(img);

        // run throughout all templates
        for (SignTemplate template : templates) {

            // skip irrelevant traffic signs
            if (toSkip(template, shape))
                continue;

            // mask the image with the sign shape
            Mat imgMasked = new Mat();
            img.copyTo(imgMasked, template.mask);

            // change colors for better matching
            switch (colorMode) {
                case BW:
                    // color filtering
                    filterByTemplate(template, img);
                    // Template Matching
                    Imgproc.matchTemplate(img, template.bw, template.matchResult, match_method);
                    break;
                case GRAY:
                    Mat fixedGray = new Mat();
                    // convert to gray scale
                    Imgproc.cvtColor(imgMasked, fixedGray, Imgproc.COLOR_RGBA2GRAY);
                    // H_img(x)
                    Imgproc.equalizeHist(fixedGray, fixedGray);
                    // apply on image H_tpl^-1(H_img(x))
                    Core.LUT(fixedGray, template.rgb_g_LUTs.col(3).t(), fixedGray);
                    // Template Matching
                    Imgproc.matchTemplate(fixedGray, template.gray, template.matchResult, match_method);
                    break;
                case COLORS:
                    List<Mat> fixedChannels = new ArrayList<>();
                    List<Mat> templateChannels = new ArrayList<>();
                    // get the rgb channels
                    Core.split(template.img, templateChannels);
                    Core.split(imgMasked, channels);
                    // transform the image to get matching histograms -> T(x) = H_tpl^-1(H_img(x))
                    for (int i = 0; i < 3; i++) {
                        fixedChannels.add(new Mat());
                        // H_img(x)
                        Imgproc.equalizeHist(channels.get(i), fixedChannels.get(i));
                        // apply on image H_tpl^-1(H_img(x))
                        Core.LUT(fixedChannels.get(i), template.rgb_g_LUTs.col(i).t(),
                                fixedChannels.get(i));
                        // Template Matching
                        Mat result = new Mat();
                        Imgproc.matchTemplate(fixedChannels.get(i), templateChannels.get(i), result, match_method);
                        if (i > 0)
                            Core.add(result, template.matchResult, template.matchResult);
                        else
                            template.matchResult = result;
                    }
                    Core.divide(template.matchResult, new Scalar(3), template.matchResult);
                    // for debugging ----------------------------------------------------
                    channels = fixedChannels;

                    // ------------------------------------------------------------------
                    break;
            }
//            // normalize the contour to ones and zeros
//            Core.normalize(img, img, 0, 1, Core.NORM_MINMAX, -1, new Mat());
            // get result
            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(template.matchResult);
            if (matchVal < minMaxLocResult.maxVal) {
                matchVal = minMaxLocResult.maxVal;
                matchId = template.id;
            }

        }

        // template height and width
        int width = templates.get(0).gray.width();
        int height = templates.get(0).gray.height();

        // draw rectangle if template matching succeeded
        if (matchVal >= 0.82) {
            Scalar color;
            Point Start, End;
            color = new Scalar(0, 0, 255);
            Start = new Point(scale * r.x, scale * r.y);
            End = new Point(scale * (r.x + r.width), scale * (r.y + r.height));
            Imgproc.rectangle(dst, Start, End, color, 5);
            //for debugging ----------------------------------
            if(colorMode == ColorMode.COLORS) {
                Mat merged = new Mat();
                Core.merge(channels, merged);
                Size s = r.size();
                Rect r2 = new Rect((int) (scale * r.x), (int) (scale * r.y),
                        (int) (scale * r.width), (int) (scale * r.height));
                s.height = scale * s.height;
                s.width = scale * s.width;
                Imgproc.resize(merged, merged, s);
                merged.copyTo(dst.submat(r2));
            }
            // ------------------------------------------------
            ImageOperations.setLabel(dst, Integer.toString(matchId), shape.points, scale);
        } else
            ImageOperations.setLabel(dst, shapesNames[shape.name.ordinal()], shape.points, scale);
    }

    /**
     * Color filtering by a given template
     *
     * @param template for deciding which colors to filter
     * @param img      the image to filter
     */
    public static void filterByTemplate(SignTemplate template, Mat img) {

        Mat bw_1 = new Mat();
        Mat bw_2 = new Mat();

        // filter red and black
        if (template.type == SignType.PROHIBITORY || template.type == SignType.DANGER) {
            ImageOperations.colorFilter(img, bw_1, ImageOperations.ColorName.RED);
            ImageOperations.colorFilter(img, bw_2, ImageOperations.ColorName.BLACK);
            Core.addWeighted(bw_1, 1, bw_2, 1, 0, img);
        }
        // filter only white
        else if (template.type == SignType.MANDATORY)
            ImageOperations.colorFilter(img, img, ImageOperations.ColorName.WHITE);
            // filter color by id
        else if (template.type == SignType.OTHER)
            switch (template.id) {
                case 12:
                    ImageOperations.colorFilter(img, img, ImageOperations.ColorName.YELLOW);
                    break;
                case 13:
                case 14:
                case 17:
                    ImageOperations.colorFilter(img, img, ImageOperations.ColorName.WHITE);
                    break;
                default:
                    ImageOperations.colorFilter(img, img, ImageOperations.ColorName.BLACK);
            }

    }


    /**
     * decide to skip by template and shape properties
     *
     * @param template template to check
     * @param shape    shape to check
     */
    public static boolean toSkip(SignTemplate template, ImageOperations.Shape shape) {
        switch (template.type) {
            case PROHIBITORY:
                if (!((shape.color == ImageOperations.ColorName.RED ||
                        shape.color == ImageOperations.ColorName.OTHER) &&
                        shape.name == ImageOperations.ShapeName.CIRCLE))
                    return true;
                break;
            case DANGER:
                if (!((shape.color == ImageOperations.ColorName.RED ||
                        shape.color == ImageOperations.ColorName.OTHER) &&
                        shape.name == ImageOperations.ShapeName.TRIANGLE))
                    return true;
                break;
            case MANDATORY:
                if (!((shape.color == ImageOperations.ColorName.BLUE ||
                        shape.color == ImageOperations.ColorName.OTHER) &&
                        shape.name == ImageOperations.ShapeName.CIRCLE))
                    return true;
                break;
            case OTHER:
                switch (template.id) {
                    case 12:
                        if (!((shape.color == ImageOperations.ColorName.BLACK ||
                                shape.color == ImageOperations.ColorName.OTHER)
                                && shape.name == ImageOperations.ShapeName.SQUARE))
                            return true;
                        break;
                    case 13:
                        if (!((shape.color == ImageOperations.ColorName.RED ||
                                shape.color == ImageOperations.ColorName.OTHER)
                                && shape.name == ImageOperations.ShapeName.OCTAGON))
                            return true;
                        break;
                    case 14:
                        if (!((shape.color == ImageOperations.ColorName.RED ||
                                shape.color == ImageOperations.ColorName.OTHER)
                                && shape.name == ImageOperations.ShapeName.TRIANGLE))
                            return true;
                        break;
                    case 17:
                        if (!((shape.color == ImageOperations.ColorName.RED ||
                                shape.color == ImageOperations.ColorName.OTHER)
                                && shape.name == ImageOperations.ShapeName.CIRCLE))
                            return true;
                        break;
                    default:
                        if (!((shape.color == ImageOperations.ColorName.BLACK ||
                                shape.color == ImageOperations.ColorName.OTHER)
                                && shape.name == ImageOperations.ShapeName.CIRCLE))
                            return true;
                }
        }

        return false;
    }

}
