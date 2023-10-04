import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(2)

#detector = FaceDetector()
detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)


while True:
    success, img = cap.read()

    #img, bboxs = detector.findFaces(img)
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        # Loop through each detected face
        for face in faces:
            # Get specific points for the eye
            # leftEyeUpPoint: Point above the left eye
            # leftEyeDownPoint: Point below the left eye
            leftEyeUpPoint = face[159]
            leftEyeDownPoint = face[23]

            # Calculate the vertical distance between the eye points
            # leftEyeVerticalDistance: Distance between points above and below the left eye
            # info: Additional information (like coordinates)
            leftEyeVerticalDistance, info = detector.findDistance(leftEyeUpPoint, leftEyeDownPoint)

            # Print the vertical distance for debugging or information
            print(leftEyeVerticalDistance)

    #if bboxs:
    #    for bbox in bboxs:
    #        # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
    #        center = bbox["center"]
    #       x, y, w, h = bbox['bbox']
    #       score = int(bbox['score'][0] * 100)

            # ---- Draw Data  ---- #
    #       cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
    #       cvzone.putTextRect(img, f'{score}%', (x, y - 10))
    #       cvzone.cornerRect(img, (x, y, w, h))
    
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()