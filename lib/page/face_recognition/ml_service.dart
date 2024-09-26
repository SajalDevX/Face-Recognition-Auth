import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imglib;
import '../../../models/user.dart';
import '../../utils/local_db.dart';
import '../../utils/utils.dart';
import 'image_converter.dart';

class MLService {
  late Interpreter interpreter;
  List? predictedArray;

  // Predict the user's face and return user details if matched during login
  Future<User?> predict(
      CameraImage cameraImage, Face face, bool loginUser, String name) async {
    // Preprocess the image and reshape it for the model
    List input = _preProcess(cameraImage, face);
    input = input.reshape([1, 112, 112, 3]);

    // Prepare output list of size 192 (expected output of MobileFaceNet)
    List output = List.generate(1, (index) => List.filled(192, 0));

    // Ensure the interpreter is initialized
    await initializeInterpreter();

    // Run the interpreter for prediction
    interpreter.run(input, output);
    output = output.reshape([192]);

    // Store the predicted result
    predictedArray = List.from(output);

    if (!loginUser) {
      // Save user details to local storage for future logins
      LocalDB.setUserDetails(User(name: name, array: predictedArray!));
      return null;
    } else {
      // Get the saved user details for comparison
      User? user = LocalDB.getUser();
      List userArray = user.array!;
      int minDist = 999;
      double threshold = 1.5;

      // Calculate Euclidean distance to match the faces
      var dist = euclideanDistance(predictedArray!, userArray);
      if (dist <= threshold && dist < minDist) {
        return user; // Return the matched user
      } else {
        return null; // No match found
      }
    }
  }

  // Calculate Euclidean distance between two face embeddings
  num euclideanDistance(List l1, List l2) {
    double sum = 0;
    for (int i = 0; i < l1.length; i++) {
      sum += pow((l1[i] - l2[i]), 2);
    }
    return pow(sum, 0.5);
  }

  // Initialize the TFLite interpreter with GPU delegate (if available)
  Future<void> initializeInterpreter() async {
    Delegate? delegate;
    try {
      // Initialize GPU delegate for Android or iOS
      if (Platform.isAndroid) {
        delegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
          ),
        );
      } else if (Platform.isIOS) {
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
            allowPrecisionLoss: true,
          ),
        );
      }

      var interpreterOptions = InterpreterOptions();

      // Add GPU delegate if available
      if (delegate != null) {
        interpreterOptions.addDelegate(delegate);
      }

      // Load the model from assets
      interpreter = await Interpreter.fromAsset(
        'assets/mobilefacenet.tflite',
        options: interpreterOptions,
      );

      printIfDebug("Model successfully loaded!");
    } catch (e) {
      // Handle errors related to model loading or initialization
      printIfDebug('Failed to load model.');
      printIfDebug(e);

      // Fallback to CPU if GPU delegate fails
      try {
        interpreter =
            await Interpreter.fromAsset('assets/mobilefacenet.tflite');
        printIfDebug("Model loaded on CPU as fallback.");
      } catch (e) {
        printIfDebug("Failed to load model on CPU.");
        printIfDebug(e);
      }
    }
  }

  // Pre-process the camera image by cropping and resizing the face
  List _preProcess(CameraImage image, Face faceDetected) {
    // Crop the face from the image
    imglib.Image croppedImage = _cropFace(image, faceDetected);
    // Resize the image to 112x112 for the model
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, 112);
    // Convert the image to a Float32List for input to the model
    Float32List imageAsList = _imageToByteListFloat32(img);
    return imageAsList;
  }

  // Crop the face from the CameraImage using the bounding box of the detected face
  imglib.Image _cropFace(CameraImage image, Face faceDetected) {
    imglib.Image convertedImage = _convertCameraImage(image);
    double x = faceDetected.boundingBox.left - 10.0;
    double y = faceDetected.boundingBox.top - 10.0;
    double w = faceDetected.boundingBox.width + 10.0;
    double h = faceDetected.boundingBox.height + 10.0;
    return imglib.copyCrop(
        convertedImage, x.round(), y.round(), w.round(), h.round());
  }

  // Convert CameraImage to an image format compatible with the image package
  imglib.Image _convertCameraImage(CameraImage image) {
    var img =
        convertToImage(image); // Use a helper to convert CameraImage to Image
    var img1 = imglib.copyRotate(img!, -90); // Rotate the image if necessary
    return img1;
  }

  // Convert the image to a Float32List for input to the TensorFlow model
  Float32List _imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }
}
