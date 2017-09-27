package ru.simcase.shlyamster.mystreet;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Vibrator;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static android.R.attr.bitmap;
import static org.opencv.android.CameraBridgeViewBase.*;
import static org.opencv.core.Core.FONT_HERSHEY_DUPLEX;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final String[] CASCADES       = {"crosswalk_cascade", "old_dont_parking_cascade", "main_road_cascade"};
    private static final String[] LABELS         = {"Crosswalk", "Don't Parking", "Main Road"};
    private static final Scalar[] COLORS         = {new Scalar(0, 255, 0, 255), new Scalar(255, 0, 0, 255), new Scalar(255, 255, 0, 255)};

    private static final int      RECT_THICKNESS = 5;
    private static final Scalar   FONT_COLOR     = new Scalar(255, 255, 255, 255);
    private static final int      FONT_FACE      = FONT_HERSHEY_DUPLEX;
    private static final double   FONT_SCALE     = 1;
    private static final int      FONT_THICKNESS = 1;

    private static int minNeighbors = 1;

    private Mat imageRGBA, imageGRAY;
    private ArrayList<CascadeClassifier> classifiers;
    private CameraBridgeViewBase camera;
    private Vibrator vibrator;
    private boolean renderLabels, renderInfo;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    if (classifiers.isEmpty()) {
                        for (String cascadeName : CASCADES) {
                            CascadeClassifier cascadeClassifier = loadCascadeClassifier(cascadeName);
                            classifiers.add(cascadeClassifier);
                        }
                    }
                    camera.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private CascadeClassifier loadCascadeClassifier(String cascadeName) {
        CascadeClassifier cascadeClassifier = null;
        try {
            InputStream inputStream = getResources().openRawResource(getResources().getIdentifier(cascadeName, "raw", getPackageName()));
            File dir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(dir, cascadeName + ".xml");
            FileOutputStream outputStream = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();

            cascadeClassifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
            cascadeClassifier.load(cascadeFile.getAbsolutePath());

            if (cascadeClassifier.empty()) {
                cascadeClassifier = null;
            }

            dir.delete();

        } catch (IOException exception) {
            exception.printStackTrace();
            cascadeClassifier = null;
        }

        return cascadeClassifier;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        renderLabels = false;
        renderInfo = false;

        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        classifiers = new ArrayList<>();

        camera = (CameraBridgeViewBase) findViewById(R.id.view);
        camera.setVisibility(VISIBLE);
        camera.setCvCameraViewListener(this);
        camera.setMaxFrameSize(960, 540);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (camera != null) {
            camera.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        camera.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        imageRGBA = new Mat(height, width, CvType.CV_8UC4);
        imageGRAY = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        imageRGBA.release();
        imageGRAY.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        double startFrameTime = SystemClock.currentThreadTimeMillis();

        imageRGBA = inputFrame.rgba();
        imageGRAY = inputFrame.gray();

        for (int cascadeIndex = 0; cascadeIndex < classifiers.size(); cascadeIndex++) {
            CascadeClassifier cascadeClassifier = classifiers.get(cascadeIndex);

            MatOfRect matOfRect = new MatOfRect();
            cascadeClassifier.detectMultiScale(imageGRAY, matOfRect, 1.4, minNeighbors, 2, new Size(1, 1), new Size(600, 600));

            ArrayList<Rect> rectArrayList = new ArrayList<>();
            ArrayList<Rect> addRectArrayList = new ArrayList<>();
            for (Rect rect : matOfRect.toArray()) {
                rectArrayList.add(rect);
                addRectArrayList.add(rect);
            }


            for (Rect iRect : addRectArrayList) {
                for (Rect jRect : addRectArrayList) {
                    if (!iRect.equals(jRect)) {
                        if (iRect.contains(jRect.tl()) && iRect.contains(jRect.br())) {
                            rectArrayList.remove(jRect);
                        }
                        if (jRect.contains(iRect.tl()) && jRect.contains(iRect.br())) {
                            rectArrayList.remove(iRect);
                        }
                    }
                }
            }

            String label = LABELS[cascadeIndex];
            Scalar color = COLORS[cascadeIndex];

            for (Rect rect : rectArrayList) {
                if (renderLabels) {
                    Imgproc.putText(imageRGBA, label, new Point(rect.x, rect.y - 10), FONT_FACE, FONT_SCALE, color, FONT_THICKNESS);
                }
                Imgproc.rectangle(imageRGBA, rect.tl(), rect.br(), color, RECT_THICKNESS);
            }
        }

        double endFrameTime = SystemClock.currentThreadTimeMillis();
        double frameTime = Math.abs(endFrameTime - startFrameTime);

        if (renderInfo) {
            List<MatOfPoint> pointList = new ArrayList<>();
            pointList.add(new MatOfPoint(new Point(0, 0), new Point(310, 0), new Point(310, 80), new Point(0, 80)));
            Imgproc.fillPoly(imageRGBA, pointList, new Scalar(0, 0, 0, 255));
            Imgproc.putText(imageRGBA, String.format("Min Neighbors: %d", minNeighbors), new Point(10, 30), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS);
            Imgproc.putText(imageRGBA, String.format("FPS: %.2f", 1000.0 / frameTime), new Point(10, 65), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS);
        }

        return imageRGBA;
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_UP:
                vibrator.vibrate(50L);
                minNeighbors++;
                if (minNeighbors > 20) {
                    minNeighbors = 20;
                }
                return true;
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                vibrator.vibrate(50L);
                minNeighbors--;
                if (minNeighbors < 0) {
                    minNeighbors = 0;
                }
                return true;
            default:
                return super.onKeyDown(keyCode, event);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.render_labels:
                renderLabels = !renderLabels;
                item.setChecked(renderLabels);
                return true;
            case R.id.render_info:
                renderInfo = !renderInfo;
                item.setChecked(renderInfo);
                return true;
            case R.id.exit:
                System.exit(0);
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }
}
