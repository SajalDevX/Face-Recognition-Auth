import 'package:face_detection_app/page/face_recognition/camera_page.dart';
import 'package:face_detection_app/page/login_page.dart';
import 'package:face_detection_app/screen.dart';
import 'package:face_detection_app/utils/local_db.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:camera/camera.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:uuid/uuid.dart';

// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   await Hive.initFlutter();
//   await Hive.openBox('user_uuid_box');
//   runApp(MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'Flutter Demo',
//       home: FaceScanScreen(),
//     );
//   }
// }

class FaceDetectionService {
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: true,
      enableClassification: true,
    ),
  );

  Future<String?> detectFaceAndGenerateUUID(InputImage inputImage) async {
    try {
      final List<Face> faces = await _faceDetector.processImage(inputImage);

      if (faces.isNotEmpty) {
        // Convert the face to embeddings
        final List<double> faceEmbedding = _extractFaceEmbedding(faces.first);
        final String? uuid =
            LocalStorageService().getUUIDForFace(faceEmbedding);

        if (uuid != null) {
          return uuid; // Return existing UUID
        } else {
          // Generate new UUID if no match found
          return LocalStorageService()
              .generateAndStoreUUIDForFace(faceEmbedding);
        }
      } else {
        return null; // No face detected
      }
    } catch (e) {
      print('Error detecting face: $e');
      return null;
    }
  }

  List<double> _extractFaceEmbedding(Face face) {
    // Implement your method to extract face embeddings from the detected face
    // For simplicity, return an empty list or a placeholder
    return [];
  }

  void dispose() {
    _faceDetector.close();
  }
}

class LocalStorageService {
  static const String _uuidBox = 'user_uuid_box';

  Future<Box> _openBox() async {
    return await Hive.openBox(_uuidBox);
  }

  // Retrieve stored UUID for a given face embedding
  String? getUUIDForFace(List<double> faceEmbedding) {
    final box = Hive.box(_uuidBox);
    final uuidMap = box.get('face_embeddings', defaultValue: {});
    for (final entry in uuidMap.entries) {
      if (_compareEmbeddings(faceEmbedding, entry.value)) {
        return entry.key; // Return UUID if face matches
      }
    }
    return null; // No match found
  }

  // Generate and store UUID for a new face embedding
  String generateAndStoreUUIDForFace(List<double> faceEmbedding) {
    final box = Hive.box(_uuidBox);
    String newUUID = Uuid().v4();
    final uuidMap = box.get('face_embeddings', defaultValue: {});
    uuidMap[newUUID] = faceEmbedding; // Store face embedding with UUID
    box.put('face_embeddings', uuidMap);
    return newUUID;
  }

  // Compare face embeddings (implement a suitable comparison method)
  bool _compareEmbeddings(List<double> embedding1, List<double> embedding2) {
    // Implement your face embedding comparison logic
    // For simplicity, you could use a distance threshold or another comparison metric
    return embedding1.toString() == embedding2.toString();
  }
}

Future main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  await Hive.initFlutter();
  await HiveBoxes.initialize();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) => const MaterialApp(
        debugShowCheckedModeBanner: false,
        title: "Face Auth",
        home: LoginPage(),
      );
}
