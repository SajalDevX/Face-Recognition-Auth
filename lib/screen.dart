import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:logger/logger.dart';
import 'package:face_detection_app/main.dart';

class FaceScanScreen extends StatefulWidget {
  @override
  _FaceScanScreenState createState() => _FaceScanScreenState();
}

class _FaceScanScreenState extends State<FaceScanScreen> {
  CameraController? _cameraController;
  bool _isDetecting = false; // To control the detection process
  FaceDetectionService? _faceDetectionService;
  Logger _logger = Logger();
  String? _userUUID;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _faceDetectionService = FaceDetectionService();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final camera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front);
    _cameraController = CameraController(camera, ResolutionPreset.medium);

    await _cameraController!.initialize();
    _cameraController!.startImageStream((CameraImage image) {
      if (!_isDetecting) {
        _isDetecting = true;
        _processCameraImage(image);
      }
    });

    setState(() {});
  }

  Future<void> _processCameraImage(CameraImage image) async {
    try {
      final WriteBuffer allBytes = WriteBuffer();
      for (final Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

      final Size imageSize =
          Size(image.width.toDouble(), image.height.toDouble());

      final InputImageRotation imageRotation =
          InputImageRotationValue.fromRawValue(
                  _cameraController!.description.sensorOrientation) ??
              InputImageRotation.rotation0deg;

      final InputImageFormat inputImageFormat =
          InputImageFormatValue.fromRawValue(image.format.raw) ??
              InputImageFormat.yuv_420_888;

      final planeData = image.planes.map(
        (Plane plane) {
          return InputImagePlaneMetadata(
            bytesPerRow: plane.bytesPerRow,
            height: plane.height,
            width: plane.width,
          );
        },
      ).toList();

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        inputImageData: InputImageData(
          size: imageSize,
          imageRotation: imageRotation,
          inputImageFormat: inputImageFormat,
          planeData: planeData,
        ),
      );

      final String? faceUUID =
          await _faceDetectionService!.detectFaceAndGenerateUUID(inputImage);

      if (faceUUID != null) {
        _userUUID = faceUUID;

        if (_userUUID != null) {
          _stopFaceDetection(); // Stop camera and detection when navigating
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => UserUUIDScreen(userUUID: _userUUID!),
            ),
          );
        }
      } else {
        _logger.d('No face detected');
        _isDetecting = false; // Allow next frame for detection
      }
    } catch (e) {
      _logger.e('Error processing camera image: $e');
      _isDetecting = false; // Allow next frame for detection
    }
  }

  void _stopFaceDetection() async {
    // Stop the camera stream and dispose of resources
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _faceDetectionService?.dispose(); // Dispose of the face detector
  }

  @override
  void dispose() {
    _stopFaceDetection();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      appBar: AppBar(title: Text('Face Scan')),
      body: CameraPreview(_cameraController!),
    );
  }
}

class UserUUIDScreen extends StatelessWidget {
  final String userUUID;

  UserUUIDScreen({required this.userUUID});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('User UUID')),
      body: Center(child: Text('UUID: $userUUID')),
    );
  }
}
