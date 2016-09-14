package br.com.senior.main;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.util.Random;

import javax.imageio.ImageIO;

import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

/**
 * Sample for face detection using javaCV
 * 
 * Instructions and original source code can be found at
 * 
 * https://github.com/bytedeco/javacv
 * 
 * @author cristiano.franco
 */

public class FaceDetectionDemo {
	private static BufferedImage photo;
	private static boolean captured;

	public static void main(String[] args) throws Exception {
		String classifierName = null;
		if (args.length > 0) {
			classifierName = args[0];
		} else {
			URL url = new URL(
					"https://raw.github.com/Itseez/opencv/2.4.0/data/haarcascades/haarcascade_frontalface_alt.xml");
			File file = Loader.extractResource(url, null, "classifier", ".xml");
			file.deleteOnExit();
			classifierName = file.getAbsolutePath();
		}

		Loader.load(opencv_objdetect.class);

		CvHaarClassifierCascade classifier = new CvHaarClassifierCascade(cvLoad(classifierName));
		if (classifier.isNull()) {
			System.err.println("Error loading classifier file \"" + classifierName + "\".");
			System.exit(1);
		}

		FrameGrabber grabber = FrameGrabber.createDefault(0);
		grabber.start();

		OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();

		IplImage grabbedImage = converter.convert(grabber.grab());
		int width = grabbedImage.width();
		int height = grabbedImage.height();
		IplImage grayImage = IplImage.create(width, height, IPL_DEPTH_8U, 1);
		IplImage rotatedImage = grabbedImage.clone();

		CvMemStorage storage = CvMemStorage.create();

		FrameRecorder recorder = FrameRecorder.createDefault("output.avi", width, height);

		// http://stackoverflow.com/questions/14125758/javacv-ffmpegframerecorder-properties-explanation-needed
		recorder.setVideoCodec(1);
		recorder.start();

		CanvasFrame frame = new CanvasFrame("Face Detection - POC", CanvasFrame.getDefaultGamma() / grabber.getGamma());
		
		// TODO remove code that generates window with random axis and format
		CvMat randomR = CvMat.create(3, 3), randomAxis = CvMat.create(3, 1);
		DoubleIndexer Ridx = randomR.createIndexer(), axisIdx = randomAxis.createIndexer();
		axisIdx.put(0, (Math.random() - 0.5) / 4, (Math.random() - 0.5) / 4, (Math.random() - 0.5) / 4);
		cvRodrigues2(randomAxis, randomR, null);
		double f = (width + height) / 2.0;
		Ridx.put(0, 2, Ridx.get(0, 2) * f);
		Ridx.put(1, 2, Ridx.get(1, 2) * f);
		Ridx.put(2, 0, Ridx.get(2, 0) / f);
		Ridx.put(2, 1, Ridx.get(2, 1) / f);
		System.out.println(Ridx);

		while (frame.isVisible() && (grabbedImage = converter.convert(grabber.grab())) != null) {
			cvClearMemStorage(storage);

			cvCvtColor(grabbedImage, grayImage, CV_BGR2GRAY);

			CvSeq faces = cvHaarDetectObjects(grayImage, classifier, storage, 1.1, 3,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH);
			int total = faces.total();

			if (total > 0 && !captured) {
				int nameEnding = new Random().nextInt(1000);
				System.out.println("Picture Taken!");
				captured = true;
				photo = IplImageToBufferedImage(grabbedImage);
				File outputfile = new File("image" + nameEnding + ".jpg");
				ImageIO.write(photo, "jpg", outputfile);

			}

			if (total == 0)
				captured = false;

			for (int i = 0; i < total; i++) {
				CvRect r = new CvRect(cvGetSeqElem(faces, i));
				int x = r.x(), y = r.y(), w = r.width(), h = r.height();
				cvRectangle(grabbedImage, cvPoint(x, y), cvPoint(x + w, y + h), CvScalar.GREEN, 1, CV_AA, 0);
			}

			cvWarpPerspective(grabbedImage, rotatedImage, randomR);

			Frame rotatedFrame = converter.convert(rotatedImage);
			frame.showImage(rotatedFrame);
			recorder.record(rotatedFrame);
		}
		frame.dispose();
		recorder.stop();
		grabber.stop();

		File file = new File("output.avi");
		file.delete();
	}

	public static BufferedImage IplImageToBufferedImage(IplImage src) {
		OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
		Java2DFrameConverter paintConverter = new Java2DFrameConverter();
		Frame frame = grabberConverter.convert(src);
		return paintConverter.getBufferedImage(frame, 1);
	}
}
